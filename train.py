import os
import subprocess
import time
import functools
import shutil
from datetime import datetime
from typing import Any, Dict, Sequence, Tuple, Union
import jax

# --- CONFIGURATION ---
BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

if not BUCKET_NAME:
    raise ValueError("Environment variable 'GCS_BUCKET_NAME' is not set. Check your startup script.")

GCS_BUCKET_URI = f'gs://{BUCKET_NAME}/rick_v2_checkpoints'
# ---------------------


try:
    if not jax.devices('gpu'):
        raise RuntimeError("JAX could not find any GPU devices.")
    else:
        print(f"JAX found the following devices: {jax.devices()}")

except Exception as e:
    print(f"Error checking JAX devices: {e}")
    # Fallback/Debug: print system devices
    subprocess.run(['nvidia-smi'])
    raise RuntimeError("GPU not available to JAX. Check your setup.")


if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.')


# 1. Setup EGL for Headless Rendering
print("Configuring EGL...")
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(os.path.dirname(NVIDIA_ICD_CONFIG_PATH)):
    os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)

with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")
    

os.environ['MUJOCO_GL'] = 'egl'
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# 2. Imports
from jax import numpy as jp
import numpy as np
from etils import epath
import mujoco
from mujoco import mjx
from brax import envs
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint

# 3. Clone/Setup Assets
if not os.path.exists('rick_v2'):
    print("Cloning rick_v2 repository...")
    subprocess.run(['git', 'clone', 'https://github.com/badfortrains/rick_v2'], check=True)

ROOT_RICK_PATH = epath.Path('rick_v2')
jax.config.update('jax_default_matmul_precision', 'high')

# 4. Define Environment (Your Biped Class)
class Biped(PipelineEnv):
  def __init__(
      self,
      forward_reward_weight=2.0,
      ctrl_cost_weight=0.07,
      sideways_cost_weight=0.05,
      sideways_body_cost=0.5,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.02, 0.1), 
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,
  ):
    path = ROOT_RICK_PATH / "assemblyDerived_v9.xml"
    mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # mj_model.opt.iterations = 6
    # mj_model.opt.ls_iterations = 6
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 10 # Newton converges fast, 10 is usually plenty
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
    self._body_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'body'
    )
    self._sideways_cost_weight = sideways_cost_weight
    self._sideways_body_cost = sideways_body_cost

  def reset(self, rng: jp.ndarray) -> State:
    rng, rng1, rng2, rng_goal = jax.random.split(rng, 4)
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )
    data = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)
    com_before = data0.subtree_com[self._body_idx]
    com_after = data.subtree_com[self._body_idx]
    velocity = (com_after - com_before) / self.dt
    vel_2d = velocity[:2] 
    forward_dir = jp.array([0.0, -1.0]) 
    sideways_dir = jp.array([1.0, 0.0])

    body_quat = data.xquat[self._body_idx]
    body_sideways_dir = math.rotate(jp.array([1.0, 0.0, 0.0]), body_quat)
    body_sideways_dir = body_sideways_dir[:2]
    body_sideways_dir_normalized = body_sideways_dir/ (jp.linalg.norm(body_sideways_dir) + 1e-8)
    body_sideways_speed = jp.dot(vel_2d, body_sideways_dir_normalized)
    body_cost = self._sideways_body_cost * jp.abs(body_sideways_speed)

    sideways_dir_normalized = sideways_dir/ (jp.linalg.norm(sideways_dir) + 1e-8)
    forward_dir_normalized = forward_dir / (jp.linalg.norm(forward_dir) + 1e-8)
    forward_velocity = jp.dot(vel_2d, forward_dir_normalized)
    # Define a target speed (meters/s)
    target_speed = 0.04
    #vel_dir_normalized = vel_2d / (jp.linalg.norm(vel_2d) + 1e-8)
    
    #forward_reward = self._forward_reward_weight * jp.dot(vel_dir_normalized, forward_dir_normalized) * jp.linalg.norm(vel_2d)
    # New (Laplace/Absolute): Strong gradient at 0
    # The 'sigma' divisor controls width. 0.02 means if you are 0.02 m/s away, reward drops to ~36%
    forward_reward = self._forward_reward_weight * jp.exp(-jp.abs(forward_velocity - target_speed) / 0.02)
    sideways_speed = jp.dot(vel_2d, sideways_dir_normalized)
    sideways_cost = self._sideways_cost_weight * jp.abs(sideways_speed)

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
    healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy

    joint_pos_delta = data.qpos[7:] - data0.qpos[7:]
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(joint_pos_delta))

    reward = forward_reward + healthy_reward - ctrl_cost - sideways_cost - body_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    obs = self._get_obs(data, action)
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

  def _get_obs(self, data: jax.numpy.ndarray, action: jp.ndarray) -> jp.ndarray:
    position = data.qpos
    if self._exclude_current_positions_from_observation:
      position = position[2:]
    return jp.concatenate([position, data.qvel, data.qfrc_actuator])

envs.register_environment('biped', Biped)

# 5. Training Logic
print("Initializing Environment...")
env_name = 'biped'
env = envs.get_environment(env_name)

# Local temporary path
ckpt_path = epath.Path('/tmp/rick_v2_checkpoints')
ckpt_path.mkdir(parents=True, exist_ok=True)

config = checkpoint.network_config(
    env.observation_size,
    env.action_size,
    True,
    ppo_networks.make_ppo_networks
)

def policy_params_fn(current_step, make_policy, params):
  # Save locally
  path = ckpt_path / f'{current_step}'
  checkpoint.save(path, current_step, params, config)
  print(f"Saved checkpoint to {path}")
  
  # Sync to GCS immediately (Critical for Spot Instances)
  try:
      subprocess.run(
          ['gsutil', '-m', 'cp', '-r', str(path), GCS_BUCKET_URI], 
          check=True
      )
      print(f"Synced {path} to {GCS_BUCKET_URI}")
  except Exception as e:
      print(f"Failed to sync to GCS: {e}")

# Headless progress reporter
def progress(num_steps, metrics):
    print(f"Step: {num_steps}, Reward: {metrics['eval/episode_reward']:.3f}, Std: {metrics['eval/episode_reward_std']:.3f}", flush=True)


def get_latest_checkpoint_from_gcs(gcs_uri):
    """Finds the latest checkpoint step in GCS."""
    print(f"Checking for checkpoints in {gcs_uri}...")
    try:
        # List directories in the GCS bucket
        result = subprocess.run(
            ['gsutil', 'ls', gcs_uri], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            print("No existing checkpoints found (or bucket inaccessible).")
            return None, None

        # Parse paths to find the largest integer step
        # Expected format: gs://bucket/.../1000/
        paths = result.stdout.strip().split('\n')
        checkpoints = []
        for p in paths:
            # clean trailing slash
            clean_p = p.rstrip('/')
            try:
                # Get the last segment and convert to int
                step = int(clean_p.split('/')[-1])
                checkpoints.append((step, p))
            except ValueError:
                continue
        
        if not checkpoints:
            return None, None

        # Sort by step count (ascending) and get the last one
        latest_step, latest_path = sorted(checkpoints)[-1]
        return latest_step, latest_path

    except Exception as e:
        print(f"Error checking GCS: {e}")
        return None, None

latest_step, latest_gcs_path = get_latest_checkpoint_from_gcs(GCS_BUCKET_URI)
restore_path = None

if latest_gcs_path:
    latest_gcs_path = f"{latest_gcs_path}000{latest_step}/"
    print(f"Found latest checkpoint: {latest_step}")
    
    # 2. Download to local tmp
    local_restore_dir = epath.Path(f'/tmp/rick_v2_restore/{latest_step}')
    local_restore_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {latest_gcs_path} to {local_restore_dir}...")
    subprocess.run(
        ['gsutil', '-m', 'cp', '-r', f"{latest_gcs_path}*", str(local_restore_dir)], 
        check=True
    )
    
    # 3. Set the path for Brax
    restore_path = str(local_restore_dir)
    print("Restore path set successfully.")
else:
    print("Starting fresh training run.")

print("Starting Training...")
start_time = datetime.now()

train_fn = functools.partial(
    ppo.train, 
    num_timesteps=100_000_000, 
    num_evals=30, 
    reward_scaling=1.0,     # Changed from 0.1 (0.1 is very low for standard PPO)
    episode_length=1000, 
    normalize_observations=True, 
    action_repeat=1,
    unroll_length=10,       # Lower unroll length slightly for unstable dynamics
    num_minibatches=32, 
    num_updates_per_batch=8,
    discounting=0.97,       # Lower discount (0.99 -> 0.97) focuses on immediate survival
    learning_rate=3e-4, 
    entropy_cost=0.01,      # Increased exploration
    num_envs=4096,
    batch_size=1024, 
    seed=0, 
    policy_params_fn=policy_params_fn, 
    restore_checkpoint_path=restore_path)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'Total training time: {datetime.now() - start_time}')