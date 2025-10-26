import argparse
import os
import json
import numpy as np
import time

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from snake_env import SnakeEnv

#-- Helper function to create the environment ---
def make_env(render_mode = None, reward_mode = "length", seed = 7):
    env = SnakeEnv(render_mode = render_mode, reward_mode = reward_mode, seed = seed)
    env = Monitor(env)
    return env

#-- Main function to train the entry point ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type = int, default = 200_000)
    parser.add_argument("--reward_mode", type = str, default="length", choices= ["length", "survival"])
    parser.add_argument("--seed", type = int, default = 7)
    parser.add_argument("--logdir", type = str, default = "./logs")
    parser.add_argument("--modeldir", type =str, default = "./models")
    parser.add_argument("--results", type = str, default = "./results/reward_stats.json")

    args = parser.parse_args()
    
    os.makedirs(args.logdir, exist_ok = True)
    os.makedirs(args.modeldir, exist_ok = True)
    os.makedirs(os.path.dirname(args.results), exist_ok = True)

    env = make_env(reward_mode = args.reward_mode, seed = args.seed)
    eval_env = make_env(reward_mode = args.reward_mode, seed = args.seed + 100)
    
    # --- A2c Model ---
    model = A2C(
        policy = "MlpPolicy",
        env = env,
        verbose = 1,
        seed = args.seed,
        tensorboard_log = "./tensorboard_logs/",
        learning_rate = 7e-4,
        n_steps = 5,
        gamma = 0.99,
        gae_lambda = 1.0,
        ent_coef = 0.01,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
    )

    #-- Logger setup ---
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

    #-- Training with progress bar---
    print("\n Starting A2C training...")
    start_time = time.time()
    model.learn(total_timesteps = args.timesteps, progress_bar = True, callback = callback_list)
    elapsed = time.time() - start_time
    print(f"\n Training completed in {elapsed/60:.2f} minutes.")

    #-- Save model ---
    save_name = f"a2c_snake_{args.reward_mode}"     # This is the base name
    path = os.path.join(args.modeldir, save_name)   # This is the full path without the path
    model.save(path)                                # Stable Baselines3 will add .zip
    print(f" Saved A2C model to {path}.zip")

    #-- Evaluating average reward after training ---
    print("\n Evaluating model performance...")
    episode_rewards = []
    obs, _ = eval_env.reset()
    for ep in range(10): 
        done, total_reward = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            if done or truncated:
                episode_rewards.append(total_reward)
                obs, _ = eval_env.reset()
                break

    avg_reward = np.mean(episode_rewards)
    print(f" Average reward over 10 episodes: {avg_reward:.2f}")
            
    #-- Save reward statistics ---
    results = {}
    if os.path.exists(args.results):
        with open(args.results, "r") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    # store results under the current reward mode key
    results[args.reward_mode] = {
        "average_reward": float(avg_reward),
        "episodes": len(episode_rewards),
        "timestamps": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timesteps": args.timesteps,
        "train_time_minutes": round(elapsed / 60, 2)
    }

    with open(args.results, "w") as f:
        json.dump(results, f, indent = 4)

    print(f" Saved reward results to {args.results}")
    print(json.dumps(results, indent = 4))

    #-- Close environments ---
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()