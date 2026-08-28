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
