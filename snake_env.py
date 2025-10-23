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

        prev_direction = self.direction  # store previous direction

        # Convert agent's action to direction
        if action == 0 and self.direction != 1: self.direction = 0 #up
        if action == 1 and self.direction != 0: self.direction = 1 #down
        if action == 2 and self.direction != 3: self.direction = 2 #left
        if action == 3 and self.direction != 2: self.direction = 3 #right

        #print(f"Action chosen: {action}")   # <- Add this line

        if self.direction == 0: self.snake_pos[1] -= 10
        if self.direction == 1: self.snake_pos[1] += 10
        if self.direction == 2: self.snake_pos[0] -= 10
        if self.direction == 3: self.snake_pos[0] += 10

        self.snake_body.insert(0, list(self.snake_pos))
        
        terminated = False
        reward = 0

        ate_food = self.snake_pos == self.food_pos

        terminated = (self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10 or
            self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10 or
            self.snake_pos in self.snake_body[1:])

        if self.direction != prev_direction:
            self.turnCount += 1

        
        #reward += self._turning_to_food_reward(prev_direction)
        #reward += self._axis_direction_reward()
        #reward += self._food_distance_based_reward()
        #reward += self._food_eaten_reward(ate_food)
        
        reward += self._wall_evasion_reward(prev_direction)
        #reward += self._survival_reward()
        reward += self._death_penalty(terminated)
        #reward += self._heading_toward_wall_punish()
        #reward += self._self_collision_avoidance_reward(action)
        
        reward += self._any_turn_reward(prev_direction)


        # Eat food
        if ate_food:
            self.score += 1
            self.food_pos = [random.randrange(1, self.frame_size_x//10) * 10,
                             random.randrange(1, self.frame_size_y//10) * 10]
            # No pop, snake grows
        else:
            self.snake_body.pop()

        #print(f"Turn:{self._turning_to_food_reward(prev_direction)} WallEvasion:{self._wall_evasion_reward(prev_direction)} HeadWall:{self._heading_toward_wall_punish()} Death:{self._death_penalty(terminated)} Axis:{self._axis_direction_reward()} Dist:{self._distance_based_reward()} Apple:{self._food_eaten_reward(ate_food)}")

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

    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()

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

    def _near_wall(self, margin=20):
        x, y = self.snake_pos
        near_left = x < margin
        near_right = x > self.frame_size_x - margin - 10
        near_top = y < margin
        near_bottom = y > self.frame_size_y - margin - 10
        return near_left or near_right or near_top or near_bottom

    def _axis_direction_reward(self):
        sx, sy = self.snake_pos
        fx, fy = self.food_pos
        reward = 0

        # Horizontal movement
        if sx < fx:
            if self.direction == 3:  # moving right toward food
                reward += 1
            elif self.direction == 2:  # moving left away from food
                reward -= 1
        elif sx > fx:
            if self.direction == 2:  # moving left toward food
                reward += 1
            elif self.direction == 3:  # moving right away from food
                reward -= 1

        # Vertical movement
        if sy < fy:
            if self.direction == 1:  # moving down toward food
                reward += 1
            elif self.direction == 0:  # moving up away from food
                reward -= 1
        elif sy > fy:
            if self.direction == 0:  # moving up toward food
                reward += 1
            elif self.direction == 1:  # moving down away from food
                reward -= 1

        return reward

    def _food_eaten_reward(self, ate_food):
        if ate_food and self.reward_mode == "length":
            return 5
        return 0
    
    def _turning_to_food_reward(self, prev_direction):
        if self.direction != prev_direction and self.direction == self._get_direction_to_food():
            return 0
        return 0
    
    def _wall_evasion_reward(self, prev_direction):
        if self._near_wall(margin=20) and self.direction != prev_direction:
            return 5
        return 0
    
    def _any_turn_reward(self, prev_direction):
        if self.direction != prev_direction:
            return 1
        return 0
    
    def _death_penalty(self, dead):
        return -50 if dead else 0
    
    def _survival_reward(self):
        if self.reward_mode == "length":
            return 0.001
        elif self.reward_mode == "survival":
            return 0.1
        return 0

    def _food_distance_based_reward(self):
        sx, sy = self.snake_pos
        fx, fy = self.food_pos

        dx = abs(sx - fx)
        dy = abs(sy - fy)

        half_x = self.frame_size_x // 2
        half_y = self.frame_size_y // 2

        # Otherwise: reward increases as snake gets closer to apple in both directions
        reward_x = (half_x - dx) / 100
        reward_y = (half_y - dy) / 100
        raw_distance_reward = reward_x + reward_y
    
        # First, apply lower bound (e.g. -1.5), then upper bound (e.g. 2.0)
        bounded_reward = max(-0.1, min(raw_distance_reward, 0.2))
        
        return bounded_reward
    
    def _heading_toward_wall_punish(self, margin=30):
        x, y = self.snake_pos
        reward = 0
        # Up
        if y < margin and self.direction == 0:
            reward -= 5
        # Down
        if y > self.frame_size_y - margin - 10 and self.direction == 1:
            reward -= 5
        # Left
        if x < margin and self.direction == 2:
            reward -= 5
        # Right
        if x > self.frame_size_x - margin - 10 and self.direction == 3:
            reward -= 5
        return reward

    def _self_collision_avoidance_reward(self, action):

        x, y = self.snake_pos
        # Map action to direction
        directions = {
            0: (0, -10),  # UP
            1: (0, 10),   # DOWN
            2: (-10, 0),  # LEFT
            3: (10, 0),   # RIGHT
        }
        dx, dy = directions[action]
        next_pos = [x + dx, y + dy]
        if next_pos in self.snake_body:
            # Penalize for imminent collision with self
            return -1
        else:
            # Reward for avoiding collision
            return 1

