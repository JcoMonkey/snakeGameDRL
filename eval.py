# collects rich gameplay metrics
import argparse, os, csv
import numpy as np
from stable_baselines3 import PPO
from snake_env import SnakeEnv   # updated import
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import json

def run_episode(model, reward_mode="length", render=False, seed=7):
    env = DummyVecEnv([lambda: SnakeEnv(
        render_mode="human" if render else None,
        reward_mode=reward_mode,
        seed=seed,
        curriculum=False
    )])
    env = VecTransposeImage(env)

    obs = env.reset()
    done = False
    ep_reward, steps, max_length = 0.0, 0, 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step([int(action)])
        ep_reward += float(rewards[0])
        steps += 1
        snake_length = len(env.envs[0].snake_body)
        max_length = max(max_length, snake_length)
        done = dones[0]

    info = infos[0]
    env.close()

    return {
        "reward": ep_reward,
        "score": int(info.get("score", 0)),
        "max_length": max_length,
        "steps": steps,
        "terminated": int(done),
        "time_out": int(info.get("time_out", 0)),
        "turn_count": int(info.get("turn_count", 0)),
        "wall_turn_evade": int(info.get("wall_turn_evade", 0)),
        #"avg_food_time": info.get("avg_food_time", None),

    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="models/ppo_snake_length")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--render", type=int, default=0)
    p.add_argument("--reward_mode", type=str, default="length", choices=["length", "survival"])
    p.add_argument("--json_out", type=str, default="logs/eval_metrics.json")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if not os.path.exists(args.model_path + ".zip"):
        raise FileNotFoundError(f"Model not found: {args.model_path}.zip")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    model = PPO.load(args.model_path)

    rows = []
    for ep in range(1, args.episodes + 1):
        metrics = run_episode(model, reward_mode=args.reward_mode, render=bool(args.render), seed=args.seed)
        metrics["episode"] = ep
        rows.append(metrics)

    # Summary
    mean_reward = float(np.mean([r["reward"] for r in rows]))
    std_reward  = float(np.std([r["reward"] for r in rows]))
    mean_score  = float(np.mean([r["score"] for r in rows]))
    mean_steps  = float(np.mean([r["steps"] for r in rows]))
    mean_length = float(np.mean([r["max_length"] for r in rows]))
    term_rate   = float(np.mean([r["terminated"] for r in rows]))
    mean_turns = float(np.mean([r["turn_count"] for r in rows]))
    mean_timeout = float(np.mean([r["time_out"] for r in rows]))
    mean_wall_turn_evade = float(np.mean([r["wall_turn_evade"] for r in rows]))
    mean_food_time = mean_score/mean_steps



    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean score (food eaten): {mean_score:.2f}")
    print(f"Mean max snake length: {mean_length:.2f}")
    print(f"Mean steps: {mean_steps:.2f}")
    print(f"Termination rate: {term_rate*100:.1f}%")
    print(f"time_out (truncate/non-death): {mean_timeout*100:.1f}%")
    print(f"Mean turns: {mean_turns:.2f}")
    print(f"Mean wall evade: {mean_wall_turn_evade:.2f}")
    print(f"Mean avg food time (score/timesteps): {mean_food_time:.2f}")


    # Per-episode json
    #fieldnames = ["episode","reward","score","max_length","steps","terminated","mean_timeout","turn_count", "wall_turn_evade"]
    with open(args.json_out, "w", newline="") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved metrics to {args.json_out}")

if __name__ == "__main__":
    main()
