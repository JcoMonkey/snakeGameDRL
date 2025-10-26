from stable_baselines3.common.callbacks import BaseCallback
import json

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reward_totals = {}
        self.episode_steps = 0

    def _on_step(self) -> bool:
        # Accumulate reward breakdowns
        infos = self.locals['infos']
        for info in infos:
            breakdown = info.get('reward_breakdown', None)
            if breakdown:
                for k, v in breakdown.items():
                    self.reward_totals[k] = self.reward_totals.get(k, 0) + v
        self.episode_steps += 1
        # Detect episode end (terminated or truncated)
        if infos[0].get("terminated", False) or infos[0].get("truncated", False):
            # Log to TensorBoard
            for k, v in self.reward_totals.items():
                self.logger.record(f"per_episode/{k}", v, exclude='stdout')
            self.logger.record("per_episode/steps", self.episode_steps, exclude='stdout')
            # Reset for next episode
            self.reward_totals = {}
            self.episode_steps = 0
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
        self.episode_steps = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        breakdown = info.get("reward_breakdown", {})
        for key, val in breakdown.items():
            self.episode_rewards[key] = self.episode_rewards.get(key, 0) + val
        self.episode_steps += 1

        if info.get("terminated", False) or info.get("truncated", False):
            self.episode_num += 1
            episode_dict = {
                "episode_num": self.episode_num,
                "steps": self.episode_steps,
                **self.episode_rewards
            }
            self.all_episodes.append(episode_dict)
            self.reset_episode()
        return True

    def _on_training_end(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump(self.all_episodes, f, indent=2)