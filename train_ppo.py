import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

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
    eval_env = make_env(reward_mode=args.reward_mode, seed=args.seed + 100)

    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        n_steps=1024,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef = 0.05,
    )

    new_logger = configure(args.logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,              # how often (in environment steps) to save
        save_path="./checkpoints/",  # folder to store the saved models
        name_prefix="snake_ppo",     # name given to checkpoint files
        save_replay_buffer=True,      # optional, saves replay buffer if available
        save_vecnormalize=True        # optional, saves normalization statistics
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.modeldir,       # Folder to save best model
        log_path=args.logdir,                     # Where to log info
        eval_freq=5000,                         # How often to evaluate (e.g. every 10k steps)
        deterministic=True,                       # Use deterministic actions
        render=False,                             # Do not render during eval
        n_eval_episodes=5,                        # Evaluate on 5 episodes for each checkpoint
        verbose=1
    )

    eval_checkpoint_callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=eval_callback)

    save_name = f"ppo_snake_{args.reward_mode}"   # This is the base name
    path = os.path.join(args.modeldir, save_name) # This is the full path **without .zip**
    model.save(path)                              # Stable Baselines3 will add .zip
    print(f"Saved model to {path}.zip")


    env.close()

if __name__ == "__main__":
    main()
