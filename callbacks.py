from stable_baselines3.common.callbacks import BaseCallback
import json

CUSTOM_KEYS = ["score", "turn_count", "time_out", "wall_turn_evade"]

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = {}
        self.episode_info_stats = {}
        self.episode_steps = 0

    def reset_episode(self):
        self.episode_rewards = {}
        self.episode_info_stats = {}
        self.episode_steps = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        # Accumulate reward breakdowns
        breakdown = info.get("reward_breakdown", {})
        for key, val in breakdown.items():
            self.episode_rewards[key] = self.episode_rewards.get(key, 0) + val
        # Accumulate custom info fields (last value per episode)
        for key in CUSTOM_KEYS:
            self.episode_info_stats[key] = info.get(key, 0)
        self.episode_steps += 1

        if info.get("terminated", False) or info.get("truncated", False):
            # Log reward breakdowns
            for k, v in self.episode_rewards.items():
                self.logger.record(f"ep_reward/{k}", v)
            # Log custom episode info
            for k, v in self.episode_info_stats.items():
                self.logger.record(f"ep_info/{k}", v)
            self.logger.record("ep_info/steps", self.episode_steps)
            self.reset_episode()
        return True

class RewardBreakdownJSONCallback(BaseCallback):
    def __init__(self, json_path="reward_breakdown_log.json", verbose=0):
        super().__init__(verbose)
        self.json_path = json_path
        self.all_episodes = []
        self.episode_num = 0
        self.reset_episode()

    def reset_episode(self):
        self.episode_rewards = {}
        self.episode_custom = {}
        self.episode_steps = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        breakdown = info.get("reward_breakdown", {})
        for key, val in breakdown.items():
            self.episode_rewards[key] = self.episode_rewards.get(key, 0) + val
        for k in CUSTOM_KEYS:
            self.episode_custom[k] = info.get(k, 0)
        self.episode_steps += 1
        if info.get("terminated", False) or info.get("truncated", False):
            self.episode_num += 1
            episode_dict = {
                "episode_num": self.episode_num,
                "steps": self.episode_steps,
                **self.episode_rewards,
                **self.episode_custom
            }
            self.all_episodes.append(episode_dict)
            self.reset_episode()
        return True

    def _on_training_end(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(self.all_episodes, f, indent=2)
