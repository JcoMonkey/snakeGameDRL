# Snake DRL Mini‑Project (Gymnasium + Stable‑Baselines3)

SNAKE THING I ROBBED THE FLAPPY BIRD

* Ready‑to‑run **training**, **evaluation**, and **(optional) visualization** scripts.
* Clear places to inject **faults/bugs** for testing objectives.

---

## 1) Quickstart

```bash
# 1) Create & activate a fresh venv (recommended)
python3 -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate) activate.bat

# 2) Install deps
pip install --upgrade pip
pip install gymnasium stable-baselines3[extra] pygame numpy tensorboard

# 3) Save the files below into the same folder, then train. choose mode if needed:
python train.py --timesteps 200000 --reward_mode length --seed 7

# 4) Evaluate the trained agent (renders optional)
python eval.py --episodes 10 --render 0

# (Optional) Watch a live episode with rendering, chooose mode as needed
python visualize.py --model_path models/ppo_snake_length --fps 60 
```

**Folder layout (created at runtime):**

```
./
  flappy_env.py
  train.py
  eval.py
  visualize.py
  models/
  logs/
```

---

## 2) The Environment: `flappy_env.py`

Key ideas:

* Discrete(2) actions: **0=do nothing**, **1=flap**.
* Observation = `[bird_y, bird_vel, next_pipe_x, gap_top, gap_bottom]` (all normalized to \~\[0,1]).
* Termination on **collision**; truncation on **max\_steps** (episode cap).
* Two reward modes:

  * `survival`: +0.1 per step alive, +1.0 for each pipe passed, −1.0 on crash.
  * `coverage`: encourages exploring **new vertical bands** (simple testing proxy), +0.05 for first visit to a band in the episode, +0.05 per step alive, −1.0 on crash.


> **Where to inject faults (for testing):**
>
> * Make `PIPE_GAP` sometimes too small (e.g., 60) after N frames.
> * Randomly skip pipe collision for 1 frame.
> * Flip sign of `GRAVITY` for a short burst.

## TensorBoard

1. **Make sure you logged during training**
   In the training script (`train.py`) you already had:

   ```python
   from stable_baselines3.common.logger import configure
   new_logger = configure("./logs", ["stdout", "tensorboard"])
   model.set_logger(new_logger)
   ```

   That means logs (scalars, rewards, losses, etc.) are written into the `./logs/` folder.

2. **Launch TensorBoard** from your project root:

   ```bash
   tensorboard --logdir ./logs --port 6006
   ```

   By default it serves on `http://localhost:6006`.

3. **Open in browser**
   Go to [http://localhost:6006](http://localhost:6006). You’ll see tabs like *Scalars*, *Graphs*, *Histograms*.

   * Under **Scalars**, you can track:

     * `rollout/ep_rew_mean` → average episode reward
     * `rollout/ep_len_mean` → average episode length
     * `train/policy_loss`, `train/value_loss` → training losses
     * `time/fps` → training speed

4. **Multiple runs**
   If you train with different configs (e.g., PPO vs A2C, survival vs coverage rewards), point them to different subfolders under `logs/` (e.g., `logs/ppo_survival/`, `logs/a2c_coverage/`). TensorBoard will let you overlay/compare them.



# commands
```
python train_ppo.py --timesteps 200000 --reward_mode survival --seed 7
```

```
python visualize.py --model_path models/ppo_snake_survival --fps 60
```

```
python visualize.py --model_path models/a2c_snake_survival --fps 60
```

```
python eval.py --model_path models/ppo_snake_survival --reward_mode survival --episodes 10 --render 0 --json_out logs/survival_eval.json
```
or
```
python eval.py --model_path models/ppo_snake_survival --reward_mode survival --episodes 10 --render 0 --json_out logs/survival_eval.json
```