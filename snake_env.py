import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, reward_mode="length", seed=7):
        super().__init__()
        self.frame_size_x = 720
        self.frame_size_y = 480
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.frame_size_y//10, self.frame_size_x//10, 3),
            dtype=np.uint8
        )
        self.render_mode = render_mode

        # Pygame setup only if rendering
        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption('Snake Eater')
            self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
            self.fps_controller = pygame.time.Clock()
            self.colors = {
                "black": pygame.Color(0,0,0),
                "white": pygame.Color(255,255,255),
                "green": pygame.Color(0,255,0)
            }

        self.reset()

    def reset(self, *, seed=None, options=None):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, self.frame_size_x//10) * 10,
                         random.randrange(1, self.frame_size_y//10) * 10]
        self.score = 0
        self.turnCount = 0
        self.direction = 3  # 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.done = False
        self.steps = 0
        self.prev_food_dist = self._get_food_distance()
        return self._get_obs(), {}

    def step(self, action):
        # Convert agent's action to direction
        if action == 0 and self.direction != 1: self.direction = 0
        if action == 1 and self.direction != 0: self.direction = 1
        if action == 2 and self.direction != 3: self.direction = 2
        if action == 3 and self.direction != 2: self.direction = 3

        if self.direction == 0: self.snake_pos[1] -= 10
        if self.direction == 1: self.snake_pos[1] += 10
        if self.direction == 2: self.snake_pos[0] -= 10
        if self.direction == 3: self.snake_pos[0] += 10

        self.snake_body.insert(0, list(self.snake_pos))
        
        terminated = False
        reward = 0

        # After moving snake and before setting reward
        curr_food_dist = self._get_food_distance()

        # Positive reward for getting closer
        # Negative for moving further away
        distance_change = self.prev_food_dist - curr_food_dist
        reward += distance_change * 0.5  # scale reward as needed
        self.prev_food_dist = curr_food_dist


        prev_direction = self.direction  # store previous direction

        # Convert agent's action to direction (same as your code)
        if action == 0 and self.direction != 1: self.direction = 0
        if action == 1 and self.direction != 0: self.direction = 1
        if action == 2 and self.direction != 3: self.direction = 2
        if action == 3 and self.direction != 2: self.direction = 3

        # --- Reward for turning ---
        if self.direction != prev_direction:
            reward += 2  # or however much you want
            self.turnCount += 1

        optimal_dir = self._get_direction_to_food()

        # rewards not going in the same direction AND in the direction of food
        if self.direction != prev_direction and self.direction == optimal_dir:
            reward += 1

        # Eat food
        if self.snake_pos == self.food_pos:
            self.score += 1
            self.food_pos = [random.randrange(1, self.frame_size_x//10) * 10,
                             random.randrange(1, self.frame_size_y//10) * 10]
            # No pop, snake grows
            if self.reward_mode == "length":
                # Add bonus for eating
                reward += 10   # ← set this to whatever bonus you want (e.g., 10)
        else:
            self.snake_body.pop()

        # Game over conditions
        if (self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10 or
            self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10 or
            self.snake_pos in self.snake_body[1:]):
            terminated = True
            reward -= 2
        else:
            if self.reward_mode == "length":
                reward += len(self.snake_body)
                #print("Snake length:", len(self.snake_body))
                #print("(len(self.snake_body)-3) * (len(self.snake_body)-3):",(len(self.snake_body)-3) * (len(self.snake_body)-3))
                #exponentially increase reward based on how long it is past the default 3
            elif self.reward_mode == "survival":
                reward = 0.1

        self.done = terminated
        info = {"score": self.score, "turn count": self.turnCount}
        return self._get_obs(), reward, terminated, False, info

    def render(self):
        # Only for render_mode == "human"
        if self.render_mode != "human":
            return
        self.game_window.fill(self.colors["black"])
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, self.colors["green"], pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, self.colors["white"], pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))
        pygame.display.update()
        self.fps_controller.tick(25)

    def _get_obs(self):
        # Here you can return a grid representation or positions; for RL, more info is better
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        # Custom logic for RL can be added here
        return obs
    
    def _get_food_distance(self):
        # Uses Manhattan distance: d = |x2 - x1| + |y2 - y1| 
        return abs(self.snake_pos[0] - self.food_pos[0]) + abs(self.snake_pos[1] - self.food_pos[1])
    
    def _get_direction_to_food(self):
        sx, sy = self.snake_pos
        fx, fy = self.food_pos
        # Calculate direction to food: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        dx = fx - sx
        dy = fy - sy
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # RIGHT or LEFT
        else:
            return 1 if dy > 0 else 0  # DOWN or UP


    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

    



def length(self, terminated):
    if terminated:
        return -1.0   # Penalty for dying
    return len(self.snake_body)


