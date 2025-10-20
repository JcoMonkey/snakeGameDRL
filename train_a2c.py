import argparse
import os

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from snake_env import SnakeEnv   # <-- Changed this

def make_env(render_mode=None, reward_mode = "length", seed=7):
    env = SnakeEnv(render_mode=render_mode,reward_mode = reward_mode, seed=seed)  # <-- Updated here, remove reward_mode if SnakeEnv doesn't need it
    env = Monitor(env)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--reward_mode", type=str, default="length", choices=["length", "survival"])
    parser.add_argument("--seed", type=int, default=7)
    # ... other args
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--modeldir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    env = make_env(reward_mode=args.reward_mode, seed=args.seed)

    model = A2C(
        policy="MlpPolicy",
        learning_rate = 0.0007,
        n_steps = 25,
        gamma = 0.99,
        ent_coef = 0.05,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        use_rms_prop = True,
        rms_prop_eps = 1.0e-05,
        normalize_advantage= False,
        verbose = 1
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    save_name = f"ppo_snake_{args.reward_mode}"   # This is the base name
    path = os.path.join(args.modeldir, save_name) # This is the full path **without .zip**
    model.save(path)                              # Stable Baselines3 will add .zip
    print(f"Saved model to {path}.zip")


    env.close()

if __name__ == "__main__":
    main()
