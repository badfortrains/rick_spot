import os
import subprocess
import time
import functools
import shutil
from datetime import datetime
from typing import Any, Dict, Sequence, Tuple, Union
import imageio
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
    subprocess.run(['nvidia-smi'])
    raise RuntimeError("GPU not available to JAX. Check your setup.")

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime.'
  )

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
from brax.io import mjcf, image
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint

# 3. Clone/Setup Assets
if not os.path.exists('rick_v2'):
    print("Cloning rick_v2 repository...")
    subprocess.run(['git', 'clone', 'https://github.com/badfortrains/rick_v2'], check=True)

ROOT_RICK_PATH = epath.Path('rick_v2')
jax.config.update('jax_default_matmul_precision', 'high')

class Biped(PipelineEnv):
  def __init__(
    self,
    forward_reward_weight=1.0,
    ctrl_cost_weight=0,
    action_rate_cost_weight=0.02,
    sideways_cost_weight=0.05,
    sideways_body_cost=0.5,
    orientation_cost_weight=1.0,
    healthy_reward=1.0,
    terminate_when_unhealthy=True,
    healthy_z_range=(0.02, 0.3),
    reset_noise_scale=0.002,
    action_noise_scale=0.02,  
    obs_noise_scale=0.01,     
    exclude_current_positions_from_observation=True,
    **kwargs,
  ):
    path = ROOT_RICK_PATH / "assemblyDerived_v17.xml"
    mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 10 
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 2
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self.history_len = 5
    self.action_dim = 6
    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._action_rate_cost_weight = action_rate_cost_weight
    self._orientation_cost_weight = orientation_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._action_noise_scale = action_noise_scale 
    self._obs_noise_scale = obs_noise_scale       
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )
    self._body_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'body'
    )
    self._sideways_cost_weight = sideways_cost_weight
    self._sideways_body_cost = sideways_body_cost

  def reset(self, rng: jp.ndarray) -> State:
    rng, rng1, rng2, step_key = jax.random.split(rng, 4)
    
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )
    data = self.pipeline_init(qpos, qvel)

    action_history = jp.zeros((self.history_len, self.action_dim))
    
    obs_key, _ = jax.random.split(step_key)
    obs = self._get_obs(data, jp.zeros(self.sys.nu), action_history, obs_key)
    
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_action_rate': zero, # Track smoothing penalty
        'reward_orientation': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    
    return State(
        pipeline_state=data, 
        obs=obs, 
        reward=reward, 
        done=done, 
        metrics=metrics, 
        info={
            'action_history': action_history,
            'rng': step_key 
        }
    )

  def step(self, state: State, action: jp.ndarray) -> State:
    rng = state.info['rng']
    rng, rng_act, rng_obs = jax.random.split(rng, 3)

    # 1. Action Smoothing Cost (Calculate before modifying history)
    current_history = state.info['action_history']
    last_action = current_history[-1]
    action_rate_cost = self._action_rate_cost_weight * jp.sum(jp.square(action - last_action))

    # 2. Add Noise
    noise = jax.random.normal(rng_act, action.shape) * self._action_noise_scale
    noisy_action = jp.clip(action + noise, -1.0, 1.0)

    # Map to actuator limits
    ctrl_min = self.sys.actuator_ctrlrange[:, 0]
    ctrl_max = self.sys.actuator_ctrlrange[:, 1]
    action_scale = (ctrl_max - ctrl_min) / 2.0
    action_offset = (ctrl_max + ctrl_min) / 2.0
    scaled_action = noisy_action * action_scale + action_offset

    # Update history
    new_history = jp.roll(current_history, shift=-1, axis=0)
    new_history = new_history.at[-1].set(noisy_action) 

    data0 = state.pipeline_state
    
    # Step physics
    data = self.pipeline_step(data0, scaled_action)
    
    # Kinematics
    com_before = data0.subtree_com[self._body_idx]
    com_after = data.subtree_com[self._body_idx]
    velocity = (com_after - com_before) / self.dt
    vel_2d = velocity[:2] 
    
    forward_dir = jp.array([0.0, -1.0]) 
    sideways_dir = jp.array([1.0, 0.0])

    # Linear Forward Reward
    forward_velocity = jp.dot(vel_2d, forward_dir)
    forward_reward = self._forward_reward_weight * forward_velocity

    # Sideways penalties
    sideways_speed = jp.dot(vel_2d, sideways_dir)
    sideways_cost = self._sideways_cost_weight * jp.abs(sideways_speed)

    # Orientation logic
    root_quat = data.q[3:7]
    projected_up = math.rotate(jp.array([0., 0., 1.]), root_quat)
    tilt_cost = self._orientation_cost_weight * jp.sum(jp.square(projected_up[:2]))

    # Healthy Check
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
    
    healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy

    # Control Cost 
    joint_pos_delta = data.qpos[7:] - data0.qpos[7:]
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(joint_pos_delta))

    # Total Reward (Now includes action_rate_cost)
    reward = forward_reward + healthy_reward - ctrl_cost - sideways_cost - tilt_cost - action_rate_cost
    
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    # Get Observation
    obs = self._get_obs(data, noisy_action, new_history, rng_obs)
    
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_action_rate=-action_rate_cost, # Log the cost
        reward_orientation=-tilt_cost,
        reward_alive=healthy_reward,
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    
    return state.replace(
        pipeline_state=data, 
        obs=obs, 
        reward=reward, 
        done=done, 
        info={
            **state.info, 
            'action_history': new_history,
            'rng': rng 
        }
    )

  def _get_obs(self, data: jax.numpy.ndarray, action: jp.ndarray, action_history: jp.ndarray, rng: jp.ndarray) -> jp.ndarray:
    # 1. Get Sensor Data (Realistic: only what the Pico has access to)
    gyro_readings = data.sensordata[0:3]
    accel_readings = data.sensordata[3:6] # In MuJoCo, accel sensors inherently register gravity
    
    # 2. Add Observation Noise
    # Gyro (3) + Accel (3) = 6 total
    noise = jax.random.normal(rng, (6,)) * self._obs_noise_scale
    
    gyro_readings += noise[0:3]
    accel_readings += noise[3:6]

    # 3. Get Action History 
    history_flat = action_history.flatten()

    # 4. Concatenate (No cheating with perfect quaternions!)
    return jp.concatenate([
        history_flat,                  
        gyro_readings,           
        accel_readings           
    ])

envs.register_environment('biped', Biped)

# 5. Training Logic
print("Initializing Environment...")
env_name = 'biped'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

ckpt_path = epath.Path('/tmp/rick_v2_checkpoints')
ckpt_path.mkdir(parents=True, exist_ok=True)

config = checkpoint.network_config(
    env.observation_size,
    env.action_size,
    True, # normalize_observations
    ppo_networks.make_ppo_networks
)

def render_video(params, make_policy, step_count):
    print(f"Rendering video for step {step_count}...")
    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    states = []
    for _ in range(500):
        states.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        if state.done:
            break

    frames = eval_env.render(states, width=320, height=240, camera='track')
    video_filename = f'{step_count}.mp4'
    local_video_path = ckpt_path / video_filename
    imageio.mimsave(str(local_video_path), frames, fps=30)
    print(f"Video saved locally to {local_video_path}")
    return local_video_path

def policy_params_fn(current_step, make_policy, params):
  path = ckpt_path / f'{current_step}'
  checkpoint.save(path, current_step, params, config)
  print(f"Saved checkpoint to {path}")
  
  try:
      local_vid_path = render_video(params, make_policy, current_step)
  except Exception as e:
      print(f"Video rendering failed: {e}")
      local_vid_path = None

  try:
      subprocess.run(['gsutil', '-m', 'cp', '-r', str(path), GCS_BUCKET_URI], check=True)
      if local_vid_path:
          subprocess.run(['gsutil', 'cp', str(local_vid_path), GCS_BUCKET_URI], check=True)
          print(f"Synced video to {GCS_BUCKET_URI}/{local_vid_path.name}")
      print(f"Synced checkpoint to {GCS_BUCKET_URI}")
  except Exception as e:
      print(f"Failed to sync to GCS: {e}")

def progress(num_steps, metrics):
    print(f"Step: {num_steps}, Reward: {metrics['eval/episode_reward']:.3f}, Std: {metrics['eval/episode_reward_std']:.3f}", flush=True)

def get_latest_checkpoint_from_gcs(gcs_uri):
    print(f"Checking for checkpoints in {gcs_uri}...")
    try:
        result = subprocess.run(['gsutil', 'ls', gcs_uri], capture_output=True, text=True)
        if result.returncode != 0:
            print("No existing checkpoints found (or bucket inaccessible).")
            return None, None

        paths = result.stdout.strip().split('\n')
        checkpoints = []
        for p in paths:
            clean_p = p.rstrip('/')
            try:
                step = int(clean_p.split('/')[-1])
                checkpoints.append((step, p))
            except ValueError:
                continue
        
        if not checkpoints:
            return None, None

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
    
    local_restore_dir = epath.Path(f'/tmp/rick_v2_restore/{latest_step}')
    local_restore_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {latest_gcs_path} to {local_restore_dir}...")
    subprocess.run(['gsutil', '-m', 'cp', '-r', f"{latest_gcs_path}*", str(local_restore_dir)], check=True)
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
    reward_scaling=0.1,
    episode_length=2500,
    normalize_observations=True, 
    action_repeat=1,
    unroll_length=128,
    num_minibatches=32, 
    num_updates_per_batch=8,
    discounting=0.995, 
    learning_rate=3e-4, 
    entropy_cost=1e-3, 
    num_envs=4096,
    batch_size=16384, 
    seed=0, 
    policy_params_fn=policy_params_fn, 
    restore_checkpoint_path=restore_path)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'Total training time: {datetime.now() - start_time}')