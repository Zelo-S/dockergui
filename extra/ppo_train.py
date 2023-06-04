import gymnasium as gym
from stable_baselines3 import PPO
import os

models_dir = 'models/PPO'
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2")
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
TIMESTEPS = 10000

for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f'{models_dir}/{TIMESTEPS*i}')

"""
episodes = 10

for episodes in range(episodes):
    obs = env.reset()
    done = False
    
    while not done:
        env.render()
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        done 
"""
env.close()