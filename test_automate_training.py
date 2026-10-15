"""Local orchestration checks; no cloud resources or firmware are touched."""

import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import automate_training as training


class ResumeTests(unittest.TestCase):
    def test_resume_requires_explicit_run_id(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                training._parse_local_args(["--resume"])
        self.assertEqual(error.exception.code, 2)

    def test_resume_rejects_training_changes(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                training._parse_local_args(
                    ["--resume", "--run-id=20261014-035149", "--", "--seed=2"]
                )
        self.assertEqual(error.exception.code, 2)

    def test_new_run_still_gets_timestamp(self):
        with mock.patch.object(training, "_default_run_id", return_value="new-run"):
            args = training._parse_local_args([])
        self.assertEqual(args.run_id, "new-run")
        self.assertFalse(args.resume)

    def test_local_workflow(self):
        for resume, custom_name in ((False, None), (True, None), (True, "custom-vm")):
            with self.subTest(resume=resume, custom_name=custom_name):
                self._check_local_workflow(resume, custom_name)

    def _check_local_workflow(self, resume, custom_name):
        run_id = "20261014-035149"
        uri = f"gs://rick_spot_bucket/rick-training/{run_id}"
        manifest = {
            "latest_checkpoint_uri": uri + "/training/experiment/checkpoints/100",
            "video_uri": uri + "/rollout0.mp4",
        }
        with tempfile.TemporaryDirectory() as directory:
            argv = [
                f"--run-id={run_id}",
                "--project=rick-spot",
                "--bucket=rick_spot_bucket",
                "--zone=europe-west6-b",
                f"--pico-dir={directory}",
                f"--mujoco-playground-dir={directory}",
            ]
            if resume:
                argv.append("--resume")
            if custom_name:
                argv.append(f"--instance-name={custom_name}")
            with mock.patch.multiple(
                training,
                _run=mock.DEFAULT,
                _create_instance=mock.DEFAULT,
                _wait_for_remote=mock.DEFAULT,
                _download_export_build=mock.DEFAULT,
                _ensure_instance_stopped=mock.DEFAULT,
            ) as mocks, mock.patch.object(
                training.shutil, "which", return_value="/mock/gcloud"
            ), contextlib.redirect_stdout(io.StringIO()):
                mocks["_wait_for_remote"].return_value = manifest
                mocks["_download_export_build"].return_value = Path(directory) / "firmware.uf2"
                self.assertEqual(training._local_main(argv), 0)
                args, watched_uri = mocks["_wait_for_remote"].call_args.args
                self.assertEqual(args.instance_name, custom_name or f"rick-training-{run_id}")
                self.assertEqual(args.zone, "europe-west6-b")
                self.assertEqual(watched_uri, uri)
                if resume:
                    mocks["_create_instance"].assert_not_called()
                else:
                    mocks["_create_instance"].assert_called_once_with(args, uri)
                mocks["_download_export_build"].assert_called_once_with(args, uri, manifest)
                mocks["_ensure_instance_stopped"].assert_called_once_with(args)

    def test_already_succeeded_run_does_not_need_running_instance(self):
        args = training._parse_local_args(["--resume", "--run-id=existing-run"])
        uri = "gs://bucket/rick-training/existing-run"
        manifest = {"run_id": "existing-run"}
        with mock.patch.object(
            training, "_gcs_object_exists", return_value=True
        ) as exists, mock.patch.object(
            training, "_read_gcs_json", return_value=manifest
        ) as read, mock.patch.object(training, "_instance_status") as status:
            args.instance_name = "rick-training-existing-run"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(training._wait_for_remote(args, uri), manifest)
            exists.assert_called_once_with(uri + "/status/succeeded.json")
            read.assert_called_once_with(uri + "/manifest.json")
            status.assert_not_called()


class DiagnosticsTests(unittest.TestCase):
    def test_old_run_without_diagnostics_remains_resumable(self):
        with mock.patch.object(training, "_run") as run:
            training._download_diagnostics(Path("/unused"), {})
        run.assert_not_called()

    def test_new_run_downloads_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(training, "_run") as run:
                training._download_diagnostics(root, {"diagnostics_uri": "gs://bucket/run/diagnostics"})
            self.assertTrue((root / "diagnostics").is_dir())
            run.assert_called_once_with([
                "gcloud", "storage", "rsync", "--recursive",
                "gs://bucket/run/diagnostics", str(root / "diagnostics"),
            ])

    def test_worker_uploads_diagnostics_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            run_id = "diagnostic-test"
            uri = "gs://bucket/run"
            def run(command, **kwargs):
                if "--play_only" in command:
                    self.assertIn("--diagnostics", command)
                    output = Path(command[command.index("--logdir") + 1]) / "experiment"
                    bundle = output / "diagnostics/final"
                    bundle.mkdir(parents=True)
                    (bundle / "metadata.json").write_text("{}")
                    (output / "rollout0.mp4").write_bytes(b"video")
                return mock.Mock(returncode=0)
            with mock.patch.multiple(
                training, _wait_for_gpu=mock.DEFAULT,
                _install_remote_requirements=mock.DEFAULT,
                _check_gcs_from_training_env=mock.DEFAULT,
                _prepare_cloud_trainer=mock.DEFAULT,
                _stream_command=mock.DEFAULT,
                _experiment_uri_from_log=mock.DEFAULT,
                _latest_checkpoint=mock.DEFAULT,
                _upload_json=mock.DEFAULT,
            ) as mocks, mock.patch.object(training, "DEFAULT_REMOTE_REPO", repo), mock.patch.object(
                training, "_run", side_effect=run
            ) as commands, mock.patch.object(training.subprocess, "run"):
                mocks["_prepare_cloud_trainer"].return_value = repo / "trainer.py"
                mocks["_experiment_uri_from_log"].return_value = uri + "/training/experiment"
                mocks["_latest_checkpoint"].return_value = (100, uri + "/checkpoints/100")
                self.assertEqual(training._remote_worker({
                    "run_id": run_id, "env_name": "RickJoystickFlatTerrain",
                    "gcs_run_uri": uri, "train_args": [],
                }), 0)
                train_call = mocks["_stream_command"].call_args
                self.assertIn("--impl=warp", train_call.args[0])
                self.assertEqual(train_call.kwargs["env"]["JAX_COMPILATION_CACHE_DIR"],
                                 "gs://bucket/jax-cache")
                worker_env = train_call.kwargs["env"]
                self.assertEqual(worker_env["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "none")
                autotune_cache = repo / "cache" / "xla-autotune"
                self.assertTrue(autotune_cache.is_dir())
                self.assertIn(f"--xla_gpu_per_fusion_autotune_cache_dir={autotune_cache}",
                              worker_env["XLA_FLAGS"])
                self.assertNotIn("gs://", worker_env["XLA_FLAGS"])
                render_call = next(c for c in commands.call_args_list if "--play_only" in c.args[0])
                self.assertEqual(render_call.kwargs["env"], worker_env)
                uploads = [c.args for c in mocks["_upload_json"].call_args_list]
                manifest = next(payload for payload, dest in uploads if dest.endswith("/manifest.json"))
                self.assertEqual(manifest["diagnostics_uri"], uri + "/diagnostics")
                self.assertTrue(any(c.args[0][-1] == uri + "/diagnostics"
                                    for c in commands.call_args_list))
                run_json = repo / "rendered" / run_id / "experiment/diagnostics/run.json"
                self.assertTrue(run_json.is_file())

    def test_cloud_trainer_patch_matches_current_source(self):
        # Validate the URI patches against the neighboring trainer, without
        # changing it or contacting GCS.
        source = Path(__file__).resolve().parent.parent / "mujoco_playground/learning/train_jax_ppo.py"
        if not source.exists():
            self.skipTest("Neighboring playground checkout unavailable")
        with tempfile.TemporaryDirectory() as directory:
            real_path = training.Path
            def redirected_path(value):
                if str(value).startswith("/tmp/rick-training-"):
                    return real_path(directory) / "trainer.py"
                return real_path(value)
            result = mock.Mock(stdout=str(source))
            with mock.patch.object(training, "_run", return_value=result), mock.patch.object(
                training, "Path", side_effect=redirected_path
            ):
                patched = training._prepare_cloud_trainer("test")
            contents = patched.read_text()
            compile(contents, str(patched), "exec")
            self.assertIn('if "://" not in str(ckpt_path):', contents)
            self.assertIn('save_diagnostics("final"', contents)


class PlaygroundSourceTests(unittest.TestCase):
    def test_local_source_reaches_installed_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, installed = root / "source", root / "installed"
            archive = root / "uploaded.zip"
            for name in training.PLAYGROUND_SOURCE_FILES:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# Local implementation of {name}\n")

            def cloud(command, **kwargs):
                if "-c" in command:
                    return mock.Mock(stdout=str(installed))
                if command[-1].startswith("gs://"):
                    shutil.copyfile(command[-2], archive)
                else:
                    shutil.copyfile(archive, command[-1])
                return mock.Mock(returncode=0)

            with mock.patch.object(training, "_run", side_effect=cloud):
                training._upload_playground_source(source, "gs://bucket/source.zip")
                training._install_playground_source("gs://bucket/source.zip")
            for name in training.PLAYGROUND_SOURCE_FILES:
                self.assertEqual((source / name).read_bytes(), (installed / name).read_bytes())

    def test_unexpected_archive_paths_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            installed = root / "installed"
            with zipfile.ZipFile(archive, "w") as bundle:
                for name in training.PLAYGROUND_SOURCE_FILES:
                    bundle.writestr(name, "# source\n")
                bundle.writestr("../outside.py", "raise RuntimeError()")

            def cloud(command, **kwargs):
                if "-c" in command:
                    return mock.Mock(stdout=str(installed))
                shutil.copyfile(archive, command[-1])

            with mock.patch.object(training, "_run", side_effect=cloud):
                with self.assertRaisesRegex(ValueError, "Unexpected files"):
                    training._install_playground_source("gs://bucket/source.zip")
            self.assertFalse(installed.exists())
            self.assertFalse((root / "outside.py").exists())


class WarpDriverTests(unittest.TestCase):
    def test_current_driver_keeps_standard_warp_wheel(self):
        with mock.patch.object(training, "_run", return_value=mock.Mock(stdout="580.95.05\n")) as run:
            training._ensure_warp_driver_compatibility(["uv", "pip", "install"])
        self.assertEqual(run.call_count, 1)

    def test_older_image_selects_matching_cuda12_wheel(self):
        wheel = "https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"
        with mock.patch.object(training, "_run", side_effect=[
            mock.Mock(stdout="535.104.05\n"), mock.Mock(stdout=wheel + "\n"), mock.Mock(),
        ]) as run:
            training._ensure_warp_driver_compatibility(["uv", "pip", "install"])
        self.assertEqual(run.call_args.args[0], ["uv", "pip", "install", wheel])


if __name__ == "__main__":
    unittest.main()
