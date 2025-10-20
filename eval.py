# collects rich gameplay metrics
import argparse, os, csv
import numpy as np
from stable_baselines3 import PPO
from snake_env import SnakeEnv   # updated import
import json

def run_episode(model, reward_mode="length", render=False, seed=7):
    env = SnakeEnv(render_mode="human" if render else None, reward_mode=reward_mode, seed=seed)
    obs, info = env.reset()
    done = trunc = False

    ep_reward = 0.0
    steps = 0
    max_length = 0

    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(int(action))
        ep_reward += r
        steps += 1
        snake_length = len(env.snake_body)
        max_length = max(max_length, snake_length)

    # Episode-level metrics from env info
    score = int(info.get("score", 0))
    terminated = int(done and not trunc)
    truncated = int(trunc)
    turnCount = int(info.get("turn count", 0))

    env.close()
    return {
        "reward": float(ep_reward),
        "score": score,
        "max_length": max_length,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "turn count": turnCount
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
    mean_turns = float(np.mean([r["turn count"] for r in rows]))



    print(f"Episodes: {len(rows)}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean score (food eaten): {mean_score:.2f}")
    print(f"Mean max snake length: {mean_length:.2f}")
    print(f"Mean steps: {mean_steps:.2f}")
    print(f"Termination rate (death): {term_rate*100:.1f}%")
    print(f"Mean turns: {mean_turns:.2f}")

    # Per-episode json
    fieldnames = ["episode","reward","score","max_length","steps","terminated","truncated","turn count"]
    with open(args.json_out, "w", newline="") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved metrics to {args.json_out}")

if __name__ == "__main__":
    main()
