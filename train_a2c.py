import argparse
import os

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from snake_env import SnakeEnv

def make_env(render_mode = None, reward_mode = "length", seed = 7):
    env = SnakeEnv(render_mode = render_mode, reward_mode = reward_mode, seed = seed)
    env = Monitor(env)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 200_000)
    parser.add_argument("--reward_mode", type = str, default="length", choices= ["length", "survival"])
    parser.add_argument("--seed", type = int, default = 7)
    parser.add_argument("--logdir", type = str, default = "./logs")
    parser.add_argument("--modeldir", type =str, default = "./models")

    os.makedirs(args.logdir, exist_ok = True)
    os.makedirs(args.modeldir, exist_ok = True)

    env = make_env(reward_mode = args.reward_mode, seed = args.seed)
    eval_env = make_env(reward_mode = args.reward_mode, seed = args.seed + 100)
    
    # --- A2c Model ---
    model = A2C(
        policy = "MlpPolicy",
        env = env,
        verbose = 1,
        seed = args.seed,
        tensorboard_log = args.logdir,
        learning_rate = 7e-4,
        n_steps = 5,
        gamma = 0.99,
        gae_lambda = 1.0,
        ent_coef = 0.01,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    #-- Callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq = 10000,
        save_path = "./checkpoints/",
        name_prefix = "snake_a2c",
        save_replay_buffer = True,
        save_vecnormalize = False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = args.modeldir,
        log_path = args.logdir,
        eval_freq = 5000,
        deterministic = True,
        render = False,
        n_eval_episodes = 5,
        verbose = 1
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    #-- Training ---
    model.learn(total_timesteps = args.timesteps, progress_bar = True, callback = callback_list)

    #-- Save model ---
    save_name = f"a2c_snake_{args.reward_mode}"     # This is the base name
    path = os.path.join(args.modeldir, save_name)   # This is the full path without the path
    model.save(path)                                # Stable Baselines3 will add .zip
    print(f" Saved A2C model to {path}.zip")

    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()