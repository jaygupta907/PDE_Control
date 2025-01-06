import stable_baselines3 as sb3
import numpy as np
import gymnasium as gym
import Environment
import time
from torch.utils.tensorboard import SummaryWriter
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import matplotlib.pyplot as plt
print(torch.cuda.is_available())


save_dir = "./model_6/"
os.makedirs(save_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=500,  
    save_path=save_dir,
    name_prefix="sac_navier_stokes_no_levelset"
)

reference_state =np.full((64,64),0.5)
# reference_state[:,0:32] = 0.4
# reference_state[:,32:] = 0.8
plt.figure()
plt.imshow(reference_state)
plt.colorbar()
plt.show()
env = Environment.NavierStokes(reference_state=reference_state,steps=30)
model = sb3.SAC("MlpPolicy",env=env,verbose=1,tensorboard_log="./sac_navier_stokes_tensorboard/",batch_size=16,buffer_size=1000,learning_rate=0.001,device='cpu')
model.learn(total_timesteps=100,tb_log_name="Backstepping_2D",callback=checkpoint_callback)
print("Training Done")
env.render(type='train')

# model = sb3.SAC.load("./model_6/sac_navier_stokes_no_levelset_500_steps")
# obs,_ = env.reset(seed=0.5)
# env.render(type='test')
# for i in range(100):
#     action, _states = model.predict(obs)
#     obs, rewards, terminated,truncated, info = env.step(action)
#     if terminated or truncated:
#         print("DONE")
#         break
# env.render(type='test')