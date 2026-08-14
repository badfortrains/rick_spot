import os
import subprocess
import functools
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
from brax import math
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint
import mujoco_playground
from mujoco_playground._src import mjx_env
from mujoco_playground import wrapper
from ml_collections import config_dict

# 3. Clone/Setup Assets
if not os.path.exists('rick_v2'):
    print("Cloning rick_v2 repository...")
    subprocess.run(['git', 'clone', 'https://github.com/badfortrains/rick_v2'], check=True)

ROOT_RICK_PATH = epath.Path('rick_v2')
jax.config.update('jax_default_matmul_precision', 'high')

class Biped(mjx_env.MjxEnv):
  def __init__(self, config: config_dict.ConfigDict, config_overrides: Dict[str, Any] = None):
    super().__init__(config, config_overrides)
    path = ROOT_RICK_PATH / "v3Robot_v18.xml"
    self._mj_model = mujoco.MjModel.from_xml_path(path.as_posix())
    self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    self._mj_model.opt.iterations = 10 
    self._mj_model.opt.ls_iterations = 6
    self._mjx_model = mjx.put_model(self._mj_model)

    self._history_len = 4
    self._step_frequency = 0.8
    self._action_dim = 8
    
    self._forward_reward_weight = config.forward_reward_weight
    self._action_rate_cost_weight = config.action_rate_cost_weight
    self._orientation_cost_weight = config.orientation_cost_weight
    self._sideways_cost_weight = config.sideways_cost_weight
    self._healthy_reward = config.healthy_reward
    self._terminate_when_unhealthy = config.terminate_when_unhealthy
    self._healthy_z_range = config.healthy_z_range
    self._reset_noise_scale = config.reset_noise_scale
    self._action_noise_scale = config.action_noise_scale
    self._obs_noise_scale = config.obs_noise_scale
    
    self._body_idx = mujoco.mj_name2id(
        self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'body'
    )

  @property
  def xml_path(self) -> str:
    return (ROOT_RICK_PATH / "v3Robot_v18.xml").as_posix()

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  def reset(self, rng: jp.ndarray) -> mjx_env.State:
    rng, rng1, rng2, step_key, cmd_key = jax.random.split(rng, 5)
    target_velocity = jax.random.uniform(cmd_key, minval=0.0, maxval=0.12)
    
    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self._mjx_model.qpos0 + jax.random.uniform(
        rng1, (self._mjx_model.nq,), minval=low, maxval=hi
    )
    
    # 1. Normalize the quaternion
    root_quat = qpos[3:7]
    root_quat = root_quat / jp.linalg.norm(root_quat)
    qpos = qpos.at[3:7].set(root_quat)

    qvel = jax.random.uniform(
        rng2, (self._mjx_model.nv,), minval=low, maxval=hi
    )

    data = mujoco_playground._src.mjx_env.make_data(self._mj_model, qpos=qpos, qvel=qvel)
    data = mjx.forward(self._mjx_model, data)

    action_history = jp.zeros((self._history_len, self._action_dim))
    
    # 2. Get Ground Truth Gravity in Local Frame
    inv_quat = jp.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
    gravity_world = jp.array([0.0, 0.0, -1.0])
    gravity_local = math.rotate(gravity_world, inv_quat)
    
    obs_key, noise_key = jax.random.split(step_key)
    noisy_gravity = gravity_local + jax.random.normal(noise_key, (3,)) * self._obs_noise_scale
    
    obs = self._get_obs(data, action_history, noisy_gravity, target_velocity)
    
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_action_rate': zero,
        'reward_orientation': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    
    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info={
            'action_history': action_history,
            'rng': step_key,
            'target_velocity': target_velocity
        }
    )

  def step(self, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
    rng = state.info['rng']
    rng, rng_act, rng_obs = jax.random.split(rng, 3)

    current_history = state.info['action_history']
    last_action = current_history[-1]
    action_rate_cost = self._action_rate_cost_weight * jp.sum(jp.square(action - last_action))

    noise = jax.random.normal(rng_act, action.shape) * self._action_noise_scale
    noisy_action = jp.clip(action + noise, -1.0, 1.0)

    alpha = 0.3
    smoothed_action = alpha * noisy_action + (1.0 - alpha) * last_action

    ctrl_min = self._mjx_model.actuator_ctrlrange[:, 0]
    ctrl_max = self._mjx_model.actuator_ctrlrange[:, 1]
    action_scale = (ctrl_max - ctrl_min) / 2.0
    action_offset = (ctrl_max + ctrl_min) / 2.0
    scaled_action = smoothed_action * action_scale + action_offset

    new_history = jp.roll(current_history, shift=-1, axis=0)
    new_history = new_history.at[-1].set(noisy_action) 

    data0 = state.data
    
    data = mujoco_playground._src.mjx_env.step(self._mjx_model, data0, scaled_action, self.n_substeps)
    
    root_quat = data.qpos[3:7]
    inv_quat = jp.array([root_quat[0], -root_quat[1], -root_quat[2], -root_quat[3]])
    gravity_world = jp.array([0.0, 0.0, -1.0])
    gravity_local = math.rotate(gravity_world, inv_quat)
    
    noisy_gravity = gravity_local + jax.random.normal(rng_obs, (3,)) * self._obs_noise_scale
    target_velocity = state.info['target_velocity']
    
    com_before = data0.subtree_com[self._body_idx]
    com_after = data.subtree_com[self._body_idx]
    velocity = (com_after - com_before) / self.dt
    vel_2d = velocity[:2] 
    
    forward_dir = jp.array([0.0, -1.0]) 
    forward_velocity = jp.dot(vel_2d, forward_dir)
    
    velocity_error = forward_velocity - target_velocity
    shaping_constant = 100.0 
    forward_reward = self._forward_reward_weight * jp.exp(-shaping_constant * jp.square(velocity_error))
    sideways_dir = jp.array([1.0, 0.0])

    sideways_speed = jp.dot(vel_2d, sideways_dir)
    sideways_cost = self._sideways_cost_weight * jp.abs(sideways_speed)

    projected_up = math.rotate(jp.array([0., 0., 1.]), root_quat)
    tilt_cost = self._orientation_cost_weight * jp.sum(jp.square(projected_up[:2]))

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.qpos[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.qpos[2] > max_z, 0.0, is_healthy)
    
    healthy_reward = self._healthy_reward if self._terminate_when_unhealthy else self._healthy_reward * is_healthy

    reward = forward_reward + healthy_reward - sideways_cost - tilt_cost - action_rate_cost
    
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    obs = self._get_obs(data, new_history, noisy_gravity, target_velocity)
    
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_action_rate=-action_rate_cost,
        reward_orientation=-tilt_cost,
        reward_alive=healthy_reward,
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )
    
    return state.replace(
        data=data, 
        obs=obs, 
        reward=reward, 
        done=done, 
        info={
            **state.info, 
            'action_history': new_history,
            'rng': rng 
        }
    )

  def _get_obs(self, data: mjx.Data, action_history: jp.ndarray, noisy_gravity: jp.ndarray, target_velocity: jp.ndarray) -> jp.ndarray:
    t = data.time
    
    phase_sin = jp.sin(2.0 * jp.pi * self._step_frequency * t)
    phase_cos = jp.cos(2.0 * jp.pi * self._step_frequency * t)
    clock = jp.array([phase_sin, phase_cos])

    return jp.concatenate([
        action_history.flatten(),                  
        noisy_gravity,   
        clock.flatten(),
        jp.array([target_velocity])
    ])

# 5. Training Logic
print("Initializing Environment...")
def biped_config() -> config_dict.ConfigDict:
    return config_dict.ConfigDict({
        'ctrl_dt': 0.02,
        'sim_dt': 0.002,
        'forward_reward_weight': 2.0,
        'action_rate_cost_weight': 0.4,
        'sideways_cost_weight': 0.2,
        'orientation_cost_weight': 0.2,
        'healthy_reward': 1.0,
        'terminate_when_unhealthy': True,
        'healthy_z_range': (0.05, 0.2),
        'reset_noise_scale': 0.002,
        'action_noise_scale': 0.02,
        'obs_noise_scale': 0.06,
    })

base_env = Biped(biped_config())
env = wrapper.wrap_for_brax_training(base_env, episode_length=1000, action_repeat=1)
eval_env = wrapper.wrap_for_brax_training(base_env, episode_length=1000, action_repeat=1)

ckpt_path = epath.Path('/tmp/rick_v2_checkpoints')
ckpt_path.mkdir(parents=True, exist_ok=True)

config = checkpoint.network_config(
    env.observation_size,
    env.action_size,
    True, # normalize_observations
    ppo_networks.make_ppo_networks
)

def render_video(params, make_policy, step_count, test_speed=0.10):
    print(f"Rendering video for step {step_count} at {test_speed} m/s...")
    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(base_env.reset)
    jit_step = jax.jit(base_env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    # --- INJECT CUSTOM SPEED ---
    # 1. Override the random speed stored in the info dictionary
    new_info = state.info
    new_info['target_velocity'] = jp.array(test_speed)
    
    # 2. Overwrite the very last element of the observation array 
    # (which is where we concatenated target_velocity in _get_obs)
    new_obs = state.obs.at[-1].set(test_speed) 
    
    # 3. Create a new state with the injected command
    state = state.replace(info=new_info, obs=new_obs)
    # ---------------------------
    
    states = []
    for _ in range(500):
        states.append(state)
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, action)
        if state.done:
            break

    frames = base_env.render(states, width=320, height=240, camera='track')
    
    # Update filename to reflect the speed being tested
    video_filename = f'{step_count}_speed_{test_speed}.mp4'
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
    episode_length=1000,
    normalize_observations=True, 
    action_repeat=1,
    unroll_length=64,
    num_minibatches=32,
    num_updates_per_batch=8,
    discounting=0.995, 
    learning_rate=3e-4, 
    entropy_cost=1e-3, 
    num_envs=512,
    batch_size=1024,
    seed=0, 
    policy_params_fn=policy_params_fn, 
    restore_checkpoint_path=restore_path)

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'Total training time: {datetime.now() - start_time}')