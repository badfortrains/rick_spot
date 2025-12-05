#! /bin/bash
set -e

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

# --- 3. Create and Activate Virtual Environment ---
# We create the env in /opt so it is separate from system python
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# --- 4. Setup Python Environment (Inside venv) ---
pip install --upgrade pip

# --- 5. Download Code ---
wget https://raw.githubusercontent.com/badfortrains/rick_spot/refs/heads/main/train.py
wget https://raw.githubusercontent.com/badfortrains/rick_spot/refs/heads/main/requirements.txt

# --- 6. Install Requirements ---
# Install requirements first
pip install -r requirements.txt

# Install JAX specifically after (or ensure requirements.txt doesn't overwrite it with CPU version)
pip install --upgrade "jax[cuda12]"

# --- 7. Run Training ---
echo "Starting training..."
# We utilize the python executable inside the venv explicitly
/opt/venv/bin/python train.py

echo "Training finished. Shutting down."
sudo shutdown -h now