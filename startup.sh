#! /bin/bash
set -e
export HOME=/root
export MUJOCO_GL="egl"

# Install ops-agent for GPU monitoring
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install

# --- 1. Fetch Configuration ---
TARGET_BUCKET=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/target_bucket)
export GCS_BUCKET_NAME=$TARGET_BUCKET

# --- 2. System Deps ---
apt-get update
apt-get install -y ffmpeg git build-essential wget libegl-dev python3-pip python3-venv

# The DLVM background process might still be installing drivers.
# We loop until nvidia-smi runs successfully.
echo "Checking for GPU drivers..."
while ! nvidia-smi; do
  echo "Waiting for GPU drivers to initialize..."
  sleep 10
done
echo "GPU driver detected!"

git config --system --add safe.directory '*'

# --- 3. Create and Activate Virtual Environment ---
# Install uv and create the env in /opt so it is separate from system python
pip install uv
export UV_PYTHON_INSTALL_DIR=/opt/uv_python
if [ ! -d "/opt/venv" ]; then
  uv venv --python 3.11 /opt/venv
fi
source /opt/venv/bin/activate

# --- 4. Setup Python Environment (Inside venv) ---
if [ -d "/opt/rick_spot" ]; then
  echo "Repository already exists at /opt/rick_spot, updating..."
  cd /opt/rick_spot
  git pull || true
else
  echo "Cloning repository..."
  git clone https://github.com/badfortrains/rick_spot.git /opt/rick_spot
  cd /opt/rick_spot
fi

# --- 6. Install Requirements ---
# Install requirements first
uv pip install -r requirements.txt

# Install JAX specifically after (or ensure requirements.txt doesn't overwrite it with CPU version)
uv pip install --upgrade "jax[cuda12]"

# Allow SSH users full access to the repo, venv, and python installation
chmod -R a+rwX /opt/rick_spot /opt/venv /opt/uv_python

# --- 7. Run Training ---
echo "Starting training..."
# We utilize the python executable inside the venv explicitly
/opt/venv/bin/python train.py

echo "Training finished. Shutting down."
sudo shutdown -h now