# Rick Spot training

`automate_training.py` runs the full Rick policy pipeline:

1. Create a uniquely named GCE Spot instance from `spot-updated`.
2. Install/update the training dependencies in `/opt/venv`.
3. Train `RickJoystickFlatTerrain`, writing each Orbax checkpoint directly to
   GCS.
4. Load the last complete checkpoint, render `rollout0.mp4`, and upload it.
5. Shut down the Spot instance on success or failure.
6. Download the checkpoints into the neighboring `rick_pico` repository.
7. Run `rick_pico/export_policy.py` to replace `policy_weights.h`.
8. Build `rick_pico/build/rick_v2_pico.uf2` and upload the firmware beside the
   other run artifacts.

## Prerequisites

Authenticate the local `gcloud` CLI and set the project and bucket:

```bash
export MY_PROJECT_ID="your-project-id"
export MY_BUCKET_NAME="your-bucket-name"
gcloud auth login
gcloud config set project "$MY_PROJECT_ID"
```

The VM's service account needs permission to create, list, read, and delete
objects in the bucket (for example, Storage Object Admin scoped to this
bucket). The `cloud-platform` OAuth scope does not grant IAM permissions by
itself.

The `spot-updated` machine image is expected to contain `/opt/rick_spot` and
`/opt/venv`, as in the current manual workflow. Locally, `rick_pico` and
`mujoco_playground` are expected to be siblings of this repository, and the
Pico `build` directory must have been configured once through the Pico VS Code
extension.

Before starting PPO, the worker performs a small GCS file check and an Orbax
save/restore check. This catches cloud path or permission problems before the
first expensive training checkpoint.

## Run

From this repository:

```bash
./automate_training.py
```

Pass extra `train-jax-ppo` flags after `--`:

```bash
./automate_training.py -- --num_timesteps=200000000 --seed=2
```

Common infrastructure overrides are normal options:

```bash
./automate_training.py \
  --zone=us-central1-a \
  --machine-type=g2-standard-12 \
  --source-machine-image=spot-updated
```

Every run uses a UTC timestamp as its run ID. Cloud artifacts are stored at:

```text
gs://$MY_BUCKET_NAME/rick-training/<run-id>/
├── manifest.json
├── rollout0.mp4
├── rick_v2_pico.uf2
├── training.log
├── status/
└── training/<mujoco-run-name>/checkpoints/
```

The corresponding local checkpoints are stored in
`../rick_pico/checkpoints/<run-id>/checkpoints/`.

If the local script is interrupted, resume watching the same run with its
original run ID, project, bucket, and zone:

```bash
./automate_training.py \
  --resume \
  --run-id=20261014-035149 \
  --project=rick-spot \
  --bucket=rick_spot_bucket \
  --zone=europe-west6-b
```

This skips VM creation, waits for the existing job, then downloads checkpoints,
exports the policy, and builds/uploads the firmware as usual. It also works if
the remote job has already succeeded and shut down. If the original run used
a custom instance name, pass the same `--instance-name`. Training flags after
`--` cannot be changed when resuming.

The instance is stopped, not deleted, so its serial log remains available. A
new default instance name is generated for each run. To inspect a running or
failed startup job:

```bash
gcloud compute instances get-serial-port-output INSTANCE_NAME \
  --project="$MY_PROJECT_ID" \
  --zone=us-central1-a
```

If the Spot VM is preempted, any checkpoint whose `commit_success.txt` was
written remains in GCS. The local command exits without exporting/building an
incomplete run and prints its GCS prefix.


## Rollout diagnostics

New runs evaluate the final checkpoint with deterministic actions and fixed
reset seeds 0, 1, 2, and 3. `rollout0.mp4` is rendered from the seed-0 trace.
The worker uploads `diagnostics/` before shutting down and the local workflow
downloads it into `../rick_pico/checkpoints/<run-id>/diagnostics/`.
Resuming older runs without this artifact still works.

`diagnostics/final/` contains `traces.npz`, `summary.json`, per-seed PNG plots,
`metadata.json`, a nominal self-contained `model.mjb`, XML/mesh assets, source
snapshots, and `rscope/` playback files. The outer directory also records the
training command, checkpoint ID, training evaluation history (if available),
and the orchestration script used by the worker.

Open a bundle on macOS from the playground environment (install its
`diagnostics` extra first):

```bash
mjpython -m learning.view_diagnostics ../rick_pico/checkpoints/<run-id>/diagnostics/final
```

Read the numeric data without pickle:

```python
import numpy as np
trace = np.load("diagnostics/final/traces.npz", allow_pickle=False)
valid = trace["trace/valid"][0]
left_height = trace["trace/metrics/left_foot_height"][0][valid]
```

NPZ arrays have axes `(seed, time, ...)`. Actions and `obs` are pre-step;
`next_obs`, metrics, data, and actuator `info` are post-step. `valid` includes
the first terminal transition and excludes frozen padding. There is no
automatic reset. Rscope replays the nominal model, so use saved measurements
and sampled parameters for quantitative contact/force analysis.

For intermediate checkpoint bundles, add `--rscope_envs=4` to the training
flags. Those bundles are stored under the training experiment's `diagnostics/`
prefix in GCS; they are separate from the final bundle downloaded automatically.
This adds evaluation and serialization work at each checkpoint callback.
The recorder supports Rick's internal randomization; external wrapper domain
randomization and vision environments are rejected explicitly.

Both the playground changes and these orchestration changes must be available
to the VM: playground is installed from the remote `rick` branch, not copied
from your local working tree. No new training run is launched by installing
this instrumentation.
