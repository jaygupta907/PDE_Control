import stable_baselines3 as sb3
import numpy as np
import gymnasium as gym
import Environment
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
import os

print(torch.cuda.is_available())


save_dir = "./model/"
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=100,  
    save_path=save_dir,
    name_prefix="ppo_navier_stokes_no_levelset"
)

reference_state =np.full((64,64),0.5)
env = Environment.NavierStokes(reference_state=reference_state,steps=10)
model = sb3.PPO("MlpPolicy",env=env,verbose=1,tensorboard_log="./ppo_navier_stokes_tensorboard/",batch_size=16,learning_rate=0.001,device='cpu')
model.learn(total_timesteps=1000,tb_log_name="Backstepping_2D")
model = sb3.PPO.load("ppo_navier_stokes_no_levelset_1000")
print("Training Done")
env.render(type='train')
obs,_ = env.reset(seed=0.2)
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, terminated,truncated, info = env.step(action)
    if terminated or truncated:
        print("DONE")
        break
env.render(type='test')