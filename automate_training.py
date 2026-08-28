#!/usr/bin/env python3
"""Train Rick on a GCE Spot VM, export the policy, and build Pico firmware.

Run this file locally.  The same file is attached to the VM as its startup
script and detects the worker configuration through GCE instance metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request


METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
)
METADATA_HEADER = {"Metadata-Flavor": "Google"}
REMOTE_CONFIG_KEY = "rick-training-config"
DEFAULT_ENV_NAME = "RickJoystickFlatTerrain"
DEFAULT_REMOTE_REPO = Path("/opt/rick_spot")
DEFAULT_VENV = Path("/opt/venv")
TERMINAL_INSTANCE_STATES = {"STOPPED", "SUSPENDED", "TERMINATED"}
RESERVED_TRAIN_FLAGS = (
    "--env_name",
    "--env-name",
    "--load_checkpoint_path",
    "--load-checkpoint-path",
    "--logdir",
    "--num_videos",
    "--num-videos",
    "--play_only",
    "--play-only",
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _print_command(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    _print_command(command)
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def _metadata(key: str, timeout: float = 1.0) -> str | None:
    request = urllib.request.Request(METADATA_URL + key, headers=METADATA_HEADER)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return None


def _write_json_temp(payload: dict[str, object]) -> Path:
    file_descriptor, name = tempfile.mkstemp(prefix="rick-training-", suffix=".json")
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    return Path(name)


def _upload_json(payload: dict[str, object], destination: str) -> None:
    source = _write_json_temp(payload)
    try:
        _run(["gcloud", "storage", "cp", str(source), destination])
    finally:
        source.unlink(missing_ok=True)


def _wait_for_gpu() -> None:
    print("Waiting for the NVIDIA driver...", flush=True)
    for _ in range(60):
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            _run(["nvidia-smi"])
            return
        time.sleep(10)
    raise RuntimeError("nvidia-smi did not become ready within 10 minutes")


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    for candidate in (Path("/usr/local/bin/uv"), Path("/root/.local/bin/uv")):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("uv is not installed on the training image")


def _install_remote_requirements() -> None:
    python = DEFAULT_VENV / "bin/python"
    requirements = DEFAULT_REMOTE_REPO / "requirements.txt"
    if not python.is_file():
        raise FileNotFoundError(f"Training Python does not exist: {python}")
    if not requirements.is_file():
        raise FileNotFoundError(f"Requirements file does not exist: {requirements}")

    uv = _find_uv()
    common = [uv, "pip", "install", "--upgrade", "--python", str(python)]
    _run([*common, "--requirement", str(requirements), "gcsfs"])
    _run([*common, "jax[cuda12]"])


def _check_gcs_from_training_env(gcs_run_uri: str) -> None:
    python = DEFAULT_VENV / "bin/python"
    probe = gcs_run_uri + "/status/storage-probe.txt"
    code = """
import sys
from etils import epath

path = epath.Path(sys.argv[1])
path.write_text("ok\\n")
if path.read_text() != "ok\\n":
  raise RuntimeError("GCS probe contents did not match")
path.unlink()
"""
    _run([str(python), "-c", code, probe])


def _stream_command(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> None:
    _print_command(command)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _experiment_uri_from_log(log_path: Path, training_root: str) -> str:
    pattern = re.compile(r"^Logs are being stored in:\s*(.+?)\s*$")
    matches: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            matches.append(match.group(1))
    if not matches:
        raise RuntimeError("Could not find the cloud log directory in training output")
    experiment_uri = matches[-1].rstrip("/")
    if not experiment_uri.startswith(training_root.rstrip("/") + "/"):
        raise RuntimeError(f"Unexpected training output path: {experiment_uri}")
    return experiment_uri


def _latest_checkpoint(checkpoint_uri: str) -> tuple[int, str]:
    result = _run(
        ["gcloud", "storage", "ls", checkpoint_uri + "/*/commit_success.txt"],
        capture_output=True,
    )
    candidates: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        marker = line.strip()
        match = re.search(r"/(\d+)/commit_success\.txt$", marker)
        if match:
            candidates.append((int(match.group(1)), marker.rsplit("/", 1)[0]))
    if not candidates:
        raise RuntimeError(f"No complete checkpoints found under {checkpoint_uri}")
    return max(candidates)


def _remote_worker(config: dict[str, object]) -> int:
    run_id = str(config["run_id"])
    env_name = str(config["env_name"])
    gcs_run_uri = str(config["gcs_run_uri"]).rstrip("/")
    train_args = [str(arg) for arg in config.get("train_args", [])]
    training_root = gcs_run_uri + "/training"
    training_log = Path(f"/tmp/rick-training-{run_id}.log")
    render_root = DEFAULT_REMOTE_REPO / "rendered" / run_id
    succeeded = False

    try:
        _wait_for_gpu()
        _install_remote_requirements()
        _check_gcs_from_training_env(gcs_run_uri)
        _upload_json(
            {
                "env_name": env_name,
                "run_id": run_id,
                "started_at": _now(),
                "state": "training",
            },
            gcs_run_uri + "/status/started.json",
        )

        worker_env = os.environ.copy()
        worker_env.update(
            {
                "MUJOCO_GL": "egl",
                "PYTHONUNBUFFERED": "1",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        )
        trainer = DEFAULT_VENV / "bin/train-jax-ppo"
        if not trainer.is_file():
            raise FileNotFoundError(f"Training entry point does not exist: {trainer}")

        train_command = [
            str(trainer),
            "--env_name",
            env_name,
            "--logdir",
            training_root,
            *train_args,
            # Training checkpoints go to GCS. Rendering is done separately to
            # a local path because ffmpeg requires a normal filesystem path.
            "--num_videos=0",
        ]
        _stream_command(
            train_command,
            cwd=DEFAULT_REMOTE_REPO,
            env=worker_env,
            log_path=training_log,
        )

        experiment_uri = _experiment_uri_from_log(training_log, training_root)
        checkpoint_uri = experiment_uri + "/checkpoints"
        latest_step, latest_checkpoint_uri = _latest_checkpoint(checkpoint_uri)
        _upload_json(
            {
                "checkpoint_uri": checkpoint_uri,
                "completed_at": _now(),
                "latest_checkpoint_uri": latest_checkpoint_uri,
                "latest_step": latest_step,
                "run_id": run_id,
                "state": "rendering",
            },
            gcs_run_uri + "/status/training-complete.json",
        )

        render_root.mkdir(parents=True, exist_ok=True)
        render_command = [
            str(trainer),
            "--env_name",
            env_name,
            "--play_only",
            "--load_checkpoint_path",
            checkpoint_uri,
            "--logdir",
            str(render_root),
            "--num_videos=1",
        ]
        _run(render_command, cwd=DEFAULT_REMOTE_REPO, env=worker_env)
        videos = sorted(render_root.rglob("rollout0.mp4"))
        if len(videos) != 1:
            raise RuntimeError(
                f"Expected one rendered rollout0.mp4, found {len(videos)}"
            )

        video_uri = gcs_run_uri + "/rollout0.mp4"
        training_log_uri = gcs_run_uri + "/training.log"
        _run(["gcloud", "storage", "cp", str(videos[0]), video_uri])
        _run(["gcloud", "storage", "cp", str(training_log), training_log_uri])
        manifest = {
            "checkpoint_uri": checkpoint_uri,
            "completed_at": _now(),
            "env_name": env_name,
            "experiment_uri": experiment_uri,
            "latest_checkpoint_uri": latest_checkpoint_uri,
            "latest_step": latest_step,
            "run_id": run_id,
            "training_log_uri": training_log_uri,
            "video_uri": video_uri,
        }
        _upload_json(manifest, gcs_run_uri + "/manifest.json")
        _upload_json(
            {"completed_at": _now(), "run_id": run_id, "state": "succeeded"},
            gcs_run_uri + "/status/succeeded.json",
        )
        succeeded = True
        return 0
    except BaseException as error:  # Include startup termination signals in logs.
        traceback.print_exc()
        try:
            if training_log.is_file():
                _run(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        str(training_log),
                        gcs_run_uri + "/training.log",
                    ],
                    check=False,
                )
            _upload_json(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "failed_at": _now(),
                    "run_id": run_id,
                    "state": "failed",
                },
                gcs_run_uri + "/status/failed.json",
            )
        except BaseException:
            traceback.print_exc()
        return 1
    finally:
        state = "success" if succeeded else "failure"
        print(f"Remote workflow finished with {state}; shutting down.", flush=True)
        subprocess.run(["shutdown", "-h", "now"], check=False)


def _default_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _instance_name(run_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", run_id.lower()).strip("-")
    cleaned = cleaned[:42].rstrip("-")
    if not cleaned:
        cleaned = "run"
    return f"rick-training-{cleaned}"


def _parse_local_args(argv: list[str]) -> argparse.Namespace:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("MY_PROJECT_ID"))
    parser.add_argument("--bucket", default=os.environ.get("MY_BUCKET_NAME"))
    parser.add_argument("--zone", default=os.environ.get("RICK_TRAINING_ZONE", "us-central1-a"))
    parser.add_argument("--machine-type", default="g2-standard-12")
    parser.add_argument("--source-machine-image", default="spot-updated")
    parser.add_argument("--instance-name")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--pico-dir", type=Path, default=workspace / "rick_pico")
    parser.add_argument(
        "--mujoco-playground-dir",
        type=Path,
        default=workspace / "mujoco_playground",
    )
    parser.add_argument(
        "--export-python",
        type=Path,
        help="Python with JAX and Orbax; otherwise use mujoco_playground/.venv or uv",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra train-jax-ppo flags after --",
    )
    args = parser.parse_args(argv)
    if args.train_args[:1] == ["--"]:
        args.train_args = args.train_args[1:]
    return args


def _validate_local_args(args: argparse.Namespace) -> None:
    if not args.project:
        raise ValueError("Set MY_PROJECT_ID or pass --project")
    if not args.bucket:
        raise ValueError("Set MY_BUCKET_NAME or pass --bucket")
    args.bucket = args.bucket.removeprefix("gs://").rstrip("/")
    if "/" in args.bucket:
        raise ValueError("--bucket must be a bucket name, not a path")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise ValueError("--run-id may contain only letters, numbers, '.', '_', and '-'")
    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    for argument in args.train_args:
        if any(
            argument == flag or argument.startswith(flag + "=")
            for flag in RESERVED_TRAIN_FLAGS
        ):
            raise ValueError(f"The automation owns training flag {argument!r}")
    for directory, label in (
        (args.pico_dir, "rick_pico"),
        (args.mujoco_playground_dir, "mujoco_playground"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {directory}")


def _instance_status(args: argparse.Namespace) -> str | None:
    result = _run(
        [
            "gcloud",
            "compute",
            "instances",
            "describe",
            args.instance_name,
            "--project",
            args.project,
            "--zone",
            args.zone,
            "--format=value(status)",
        ],
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _gcs_object_exists(uri: str) -> bool:
    result = _run(
        ["gcloud", "storage", "ls", uri],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _create_instance(args: argparse.Namespace, gcs_run_uri: str) -> None:
    if _instance_status(args) is not None:
        raise RuntimeError(
            f"Instance {args.instance_name!r} already exists in {args.zone}; "
            "choose --instance-name or remove the old instance"
        )

    config_path = _write_json_temp(
        {
            "env_name": args.env_name,
            "gcs_run_uri": gcs_run_uri,
            "run_id": args.run_id,
            "train_args": args.train_args,
        }
    )
    try:
        script_path = Path(__file__).resolve()
        _run(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                args.instance_name,
                "--project",
                args.project,
                "--zone",
                args.zone,
                "--machine-type",
                args.machine_type,
                "--provisioning-model=SPOT",
                "--maintenance-policy=TERMINATE",
                "--instance-termination-action=STOP",
                "--source-machine-image",
                args.source_machine_image,
                "--scopes=https://www.googleapis.com/auth/cloud-platform",
                "--metadata=install-nvidia-driver=True",
                "--metadata-from-file",
                f"startup-script={script_path},{REMOTE_CONFIG_KEY}={config_path}",
                "--labels=purpose=rick-training",
            ]
        )
    finally:
        config_path.unlink(missing_ok=True)


def _read_gcs_json(uri: str) -> dict[str, object]:
    result = _run(
        ["gcloud", "storage", "cat", uri], capture_output=True
    )
    return json.loads(result.stdout)


def _wait_for_remote(args: argparse.Namespace, gcs_run_uri: str) -> dict[str, object]:
    success_uri = gcs_run_uri + "/status/succeeded.json"
    failed_uri = gcs_run_uri + "/status/failed.json"
    print(
        "Training is running. Startup and training logs: "
        f"gcloud compute instances get-serial-port-output {args.instance_name} "
        f"--project={args.project} --zone={args.zone}",
        flush=True,
    )
    while True:
        if _gcs_object_exists(success_uri):
            return _read_gcs_json(gcs_run_uri + "/manifest.json")
        if _gcs_object_exists(failed_uri):
            failure = _read_gcs_json(failed_uri)
            raise RuntimeError(str(failure.get("error", "Remote workflow failed")))

        status = _instance_status(args)
        if status in TERMINAL_INSTANCE_STATES:
            raise RuntimeError(
                f"Instance stopped with status {status} before success. "
                f"Any complete checkpoints remain at {gcs_run_uri}/training"
            )
        if status is None:
            raise RuntimeError("The training instance disappeared before success")
        print(f"[{_now()}] instance={status}; waiting...", flush=True)
        time.sleep(args.poll_seconds)


def _export_policy(
    args: argparse.Namespace, latest_checkpoint: Path
) -> Path:
    exporter = args.pico_dir / "export_policy.py"
    output = args.pico_dir / "policy_weights.h"
    if args.export_python:
        command = [str(args.export_python), str(exporter), str(latest_checkpoint)]
    else:
        venv_python = args.mujoco_playground_dir / ".venv/bin/python"
        if venv_python.is_file():
            command = [str(venv_python), str(exporter), str(latest_checkpoint)]
        else:
            uv = shutil.which("uv")
            if not uv:
                raise FileNotFoundError(
                    "No export Python found. Install uv, create the mujoco_playground "
                    ".venv, or pass --export-python"
                )
            command = [
                uv,
                "run",
                "--project",
                str(args.mujoco_playground_dir),
                "python",
                str(exporter),
                str(latest_checkpoint),
            ]
    _run(command, cwd=args.pico_dir)
    if not output.is_file():
        raise RuntimeError(f"Policy exporter did not create {output}")
    return output


def _build_firmware(pico_dir: Path) -> Path:
    build_dir = pico_dir / "build"
    if not (build_dir / "CMakeCache.txt").is_file():
        raise FileNotFoundError(
            f"Pico build directory is not configured: {build_dir}. "
            "Configure it once with the Raspberry Pi Pico extension."
        )

    cmake = shutil.which("cmake")
    if cmake:
        _run([cmake, "--build", str(build_dir), "--parallel"])
    else:
        cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
        match = re.search(r"^CMAKE_MAKE_PROGRAM:FILEPATH=(.+)$", cache, re.MULTILINE)
        if not match or not Path(match.group(1)).is_file():
            raise FileNotFoundError(
                "cmake is not on PATH and CMAKE_MAKE_PROGRAM was not found"
            )
        _run([match.group(1), "-C", str(build_dir)])

    firmware = build_dir / "rick_v2_pico.uf2"
    if not firmware.is_file():
        raise RuntimeError(f"Firmware build did not create {firmware}")
    return firmware


def _download_export_build(
    args: argparse.Namespace, gcs_run_uri: str, manifest: dict[str, object]
) -> Path:
    checkpoint_uri = str(manifest["checkpoint_uri"])
    run_dir = args.pico_dir / "checkpoints" / args.run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            checkpoint_uri,
            str(checkpoint_dir),
        ]
    )

    complete = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir()
        and path.name.isdigit()
        and (path / "commit_success.txt").is_file()
    ]
    if not complete:
        raise RuntimeError(f"No complete local checkpoint found in {checkpoint_dir}")
    latest_checkpoint = max(complete, key=lambda path: int(path.name))
    expected_step = int(manifest["latest_step"])
    if int(latest_checkpoint.name) != expected_step:
        raise RuntimeError(
            f"Latest downloaded checkpoint is {latest_checkpoint.name}, "
            f"but the remote manifest says {expected_step}"
        )

    _export_policy(args, latest_checkpoint)
    firmware = _build_firmware(args.pico_dir)
    firmware_uri = gcs_run_uri + "/rick_v2_pico.uf2"
    _run(["gcloud", "storage", "cp", str(firmware), firmware_uri])
    _upload_json(
        {
            "finalized_at": _now(),
            "firmware_uri": firmware_uri,
            "latest_checkpoint": str(latest_checkpoint),
            "policy_header": str(args.pico_dir / "policy_weights.h"),
            "run_id": args.run_id,
            "state": "finalized",
        },
        gcs_run_uri + "/status/finalized.json",
    )
    return firmware


def _ensure_instance_stopped(args: argparse.Namespace) -> None:
    for _ in range(18):
        status = _instance_status(args)
        if status in TERMINAL_INSTANCE_STATES or status is None:
            return
        time.sleep(10)
    print("Remote shutdown is taking too long; stopping the instance.", flush=True)
    _run(
        [
            "gcloud",
            "compute",
            "instances",
            "stop",
            args.instance_name,
            "--project",
            args.project,
            "--zone",
            args.zone,
        ]
    )


def _local_main(argv: list[str]) -> int:
    args = _parse_local_args(argv)
    _validate_local_args(args)
    if not shutil.which("gcloud"):
        raise FileNotFoundError("gcloud is not on PATH")
    args.instance_name = args.instance_name or _instance_name(args.run_id)
    gcs_run_uri = f"gs://{args.bucket}/rick-training/{args.run_id}"

    _run(
        [
            "gcloud",
            "storage",
            "buckets",
            "describe",
            f"gs://{args.bucket}",
            "--project",
            args.project,
        ],
        capture_output=True,
    )
    print(f"Run artifacts: {gcs_run_uri}", flush=True)
    _create_instance(args, gcs_run_uri)
    manifest = _wait_for_remote(args, gcs_run_uri)
    firmware = _download_export_build(args, gcs_run_uri, manifest)
    _ensure_instance_stopped(args)

    print("\nRick training automation completed.")
    print(f"Checkpoint: {manifest['latest_checkpoint_uri']}")
    print(f"Video:      {manifest['video_uri']}")
    print(f"Firmware:   {firmware}")
    print(f"Cloud copy: {gcs_run_uri}/rick_v2_pico.uf2")
    print(f"Instance:   {args.instance_name} (stopped, not deleted)")
    return 0


def main() -> int:
    remote_config = _metadata(REMOTE_CONFIG_KEY)
    if remote_config is not None:
        return _remote_worker(json.loads(remote_config))
    return _local_main(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped waiting locally; the VM startup job continues independently.")
        raise SystemExit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
