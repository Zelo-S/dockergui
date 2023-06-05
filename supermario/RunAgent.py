import retro
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

model = PPO.load("./log/best_model.zip")

def main():
    env = retro.make(game="SuperMarioBros-Nes")
    env = MaxAndSkipEnv(env, 4)
    
    obs, _ = env.reset()
    print("type obs,", type(obs))
    
    terminated, truncated = False, False

    while not terminated or truncated:
        action, state = model.predict(observation=obs)
        obs, reward, terminated, truncated, info = env.step(action=action)
        env.render()

        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()