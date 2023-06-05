import retro
import gymnasium

env = retro.make(game="SuperMarioBros-Nes")
obs = env.reset()


for _ in range(100000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()