import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, reward_mode="length", seed=7, max_steps=4000, curriculum =True):
        super().__init__()
        self.frame_size_x = 300
        self.frame_size_y = 200
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.frame_size_y//10, self.frame_size_x//10, 3),
            dtype=np.uint8
        )
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.episode_counter = 0
        self.curriculum = curriculum

        self.last_reward_breakdown = {
            "survival": 0.0,
            "death_penalty": 0.0,
            "food_eaten": 0.0,
            "move_closer": 0.0,
            "move_away": 0.0,
            "total": 0.0,
        }


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
        self.snake_pos = [self.frame_size_x // 2, self.frame_size_y // 2] #150, 100
        self.snake_body = [
            [self.snake_pos[0], self.snake_pos[1]],
            [self.snake_pos[0] - 10, self.snake_pos[1]],
            [self.snake_pos[0] - 20, self.snake_pos[1]],
        ]
        self.food_pos = [200, 100]
            
        self.score = 0
        self.turnCount = 0
        self.direction = 3  # 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.done = False
        self.steps = 0
        self.wall_turn_evade = 0
        self.prev_food_dist = self._get_food_distance()
        self.straight_steps = 0

        #self.steps_since_food = 0
        #self.food_intervals = []

        # first 300 episodes is deterministic food to teach snake to eat
        # teach it to turn, don't just have it go straight
        self.episode_counter += 1
        if self.curriculum and self.episode_counter < 100:
            direction = self.direction
            if direction == 3:  # RIGHT
                # Randomly choose up or down
                if random.random() < 0.5:
                    # UP
                    self.food_pos = [self.snake_pos[0], self.snake_pos[1] - 30]
                else:
                    # DOWN
                    self.food_pos = [self.snake_pos[0], self.snake_pos[1] + 30]
            elif direction == 2:  # LEFT
                # Similarly, force turns up or down for LEFT
                if random.random() < 0.5:
                    # UP
                    self.food_pos = [self.snake_pos[0], self.snake_pos[1] - 30]
                else:
                    # DOWN
                    self.food_pos = [self.snake_pos[0], self.snake_pos[1] + 30]
            elif direction == 0:  # UP
                # Randomly choose left or right
                if random.random() < 0.5:
                    self.food_pos = [self.snake_pos[0] - 30, self.snake_pos[1]]
                else:
                    self.food_pos = [self.snake_pos[0] + 30, self.snake_pos[1]]
            elif direction == 1:  # DOWN
                # Randomly choose left or right
                if random.random() < 0.5:
                    self.food_pos = [self.snake_pos[0] - 30, self.snake_pos[1]]
                else:
                    self.food_pos = [self.snake_pos[0] + 30, self.snake_pos[1]]
        elif self.curriculum and self.episode_counter < 1000:
        # mix: 50% deterministic, 50% random. deterministic food placed farther ahead
            if random.random() < 0.5:
                direction = self.direction
                if direction == 3:  # RIGHT
                    # Randomly choose up or down
                    if random.random() < 0.5:
                        # UP
                        self.food_pos = [self.snake_pos[0] - 40, self.snake_pos[1] - 30]
                    else:
                        # DOWN
                        self.food_pos = [self.snake_pos[0] - 40 , self.snake_pos[1] + 30]
                elif direction == 2:  # LEFT
                    # Similarly, force turns up or down for LEFT
                    if random.random() < 0.5:
                        # UP
                        self.food_pos = [self.snake_pos[0] -40, self.snake_pos[1] - 30]
                    else:
                        # DOWN
                        self.food_pos = [self.snake_pos[0] - 40, self.snake_pos[1] + 30]
                elif direction == 0:  # UP
                    # Randomly choose left or right
                    if random.random() < 0.5:
                        self.food_pos = [self.snake_pos[0] - 30, self.snake_pos[1] - 40]
                    else:
                        self.food_pos = [self.snake_pos[0] + 30, self.snake_pos[1] - 40]
                elif direction == 1:  # DOWN
                    # Randomly choose left or right
                    if random.random() < 0.5:
                        self.food_pos = [self.snake_pos[0] - 30, self.snake_pos[1] - 40]
                    else:
                        self.food_pos = [self.snake_pos[0] + 30, self.snake_pos[1] - 40]
        else:
            # Full random
            self.food_pos = [random.randrange(1, self.frame_size_x // 10) * 10,
                            random.randrange(1, self.frame_size_y // 10) * 10]
            
        self.last_reward_breakdown = {
            "survival": 0.0,
            "death_penalty": 0.0,
            "food_eaten": 0.0,
            "move_closer": 0.0,
            "move_away": 0.0,
            "total": 0.0,
        }

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
        
        stepReward = 0
        terminated = False
        snake_body_turn_evade = 0
        lastFood = 0
        #divide steps by score (reate of eating)
        #turn towards food
        #turnsaverage between eating

        ate_food = self.snake_pos == self.food_pos

        terminated = (self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10 or
            self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10 or
            self.snake_pos in self.snake_body[1:])

        if self.direction != prev_direction:
            self.turnCount += 1

        if self._wall_evade_check(prev_direction):
            self.wall_turn_evade +=1

        if self.reward_mode == "survival":
            stepReward = self._survival(terminated)

        if self.reward_mode == "length":
            stepReward = self._length(terminated, ate_food, prev_direction)
        
        #self.steps_since_food += 1

        # Eat food
        if ate_food:
            self.score += 1

            #self.food_intervals.append(self.steps_since_food)
            #self.steps_since_food = 0

            self.food_pos = [random.randrange(1, self.frame_size_x//10) * 10,
                             random.randrange(1, self.frame_size_y//10) * 10]
            # No pop, snake grows
        else:
            self.snake_body.pop()

        #print(f"Turn:{self.turnCount} WallEvasion:{self._wall_evasion_reward(prev_direction)} HeadWall:{self._heading_toward_wall_punish()} Death:{self._death_penalty(terminated)} Axis:{self._axis_direction_reward()} Dist:{self._food_distance_based_reward()} Apple:{self._food_eaten_reward(ate_food)}")

        self.steps += 1  # Increment step count
        time_out = self.steps >= self.max_steps 
        terminated = terminated or time_out

        infos = {
            "score": self.score, 
            "turn_count": self.turnCount, 
            "time_out": time_out,
            "wall_turn_evade": self.wall_turn_evade,
            #"avg_food_time": avg_food_time
            "reward_breakdown": self.last_reward_breakdown.copy()
        }
        return self._get_obs(), stepReward, terminated, False, infos

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
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        
        # Add direction to food
        direction_vec = np.zeros(4)  # [up, down, left, right]
        direction_to_food = self._get_direction_to_food()
        direction_vec[direction_to_food] = 1
        # Add danger sensors
        #danger_left = int(self._will_collide(2))  # LEFT
        #danger_right = int(self._will_collide(3)) # RIGHT
        #danger_straight = int(self._will_collide(self.direction))

        #extra_features = np.array([danger_left, danger_right, danger_straight] + list(direction_vec), dtype=np.uint8)


        rows, cols, _ = obs.shape
        scale = 10  # Each grid cell is 10x10 

        # Draw walls (first and last row & column)
        obs[0, :, :] = [255, 0, 0]
        obs[-1, :, :] = [255, 0, 0]
        obs[:, 0, :] = [255, 0, 0]
        obs[:, -1, :] = [255, 0, 0]

        # Draw snake body (except head)
        for pos in self.snake_body[1:]:
            x, y = pos[0] // scale, pos[1] // scale
            if 0 <= y < rows and 0 <= x < cols:
                obs[y, x] = [0, 255, 0]

        # Draw snake head (first element of snake_body)
        x, y = self.snake_body[0][0] // scale, self.snake_body[0][1] // scale
        if 0 <= y < rows and 0 <= x < cols:
            obs[y, x] = [0, 0, 255]

        # Draw food
        fx, fy = self.food_pos[0] // scale, self.food_pos[1] // scale
        if 0 <= fy < rows and 0 <= fx < cols:
            obs[fy, fx] = [255, 255, 255]

        return obs


# danger function

    def _will_collide(self, direction):
        # Map direction to movement
        directions = {
            0: (0, -10),  # UP
            1: (0, 10),   # DOWN
            2: (-10, 0),  # LEFT
            3: (10, 0),   # RIGHT
        }
        dx, dy = directions[direction]
        next_pos = [self.snake_pos[0] + dx, self.snake_pos[1] + dy]

        # Wall collision
        if next_pos[0] < 0 or next_pos[0] > self.frame_size_x - 10:
            return True
        if next_pos[1] < 0 or next_pos[1] > self.frame_size_y - 10:
            return True

        # Self collision
        if next_pos in self.snake_body:
            return True

        return False

    
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
    
    def _wall_evade_check(self, prev_direction):

        x, y = self.snake_pos
        if prev_direction == 0 and y <= 0 and self.direction != prev_direction:  # Upper wall
            return True    
        elif prev_direction == 1 and y >= self.frame_size_y - 10 and self.direction != prev_direction:  # Lower wall
            return True
        elif prev_direction == 2 and x <= 0 and self.direction != prev_direction:  # Left wall
            return True
        elif prev_direction == 3 and x >= self.frame_size_x - 10 and self.direction != prev_direction:  # Right wall
            return True
        
        return False


    def _axis_direction_reward(self, modifier):
        sx, sy = self.snake_pos
        fx, fy = self.food_pos
        reward = 0

        # Horizontal movement
        if sx < fx:
            if self.direction == 3:  # moving right toward food
                reward += modifier
            elif self.direction == 2:  # moving left away from food
                reward -= modifier
        elif sx > fx:
            if self.direction == 2:  # moving left toward food
                reward += modifier
            elif self.direction == 3:  # moving right away from food
                reward -= modifier

        # Vertical movement
        if sy < fy:
            if self.direction == 1:  # moving down toward food
                reward += modifier
            elif self.direction == 0:  # moving up away from food
                reward -= modifier
        elif sy > fy:
            if self.direction == 0:  # moving up toward food
                reward += modifier
            elif self.direction == 1:  # moving down away from food
                reward -= modifier

        return reward

    def _food_eaten_reward(self, ate_food, modifier):
        if ate_food:
            return modifier
        return 0
    
    def _turning_to_food_reward(self, prev_direction, modifier):
        if self.direction != prev_direction and self.direction == self._get_direction_to_food():
            return modifier
        return 0
    
    def _wall_evasion_reward(self, prev_direction, modifier):
        if self._wall_evade_check(prev_direction):
            return modifier
        else:
            return 0
    
    def _any_turn_reward(self, prev_direction):
        if self.direction != prev_direction:
            return -0.01
        return 0
    
    def _death_penalty(self, dead, modifier):
        reward = 0

        if dead:
            reward -= modifier
        
        return reward

    
    def _survival_reward(self, dead, modifier):
        reward = 0

        if not dead:
            return modifier
        
        return reward

    def _food_distance_based_reward(self, upBound, lowBound):
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
        bounded_reward = max(lowBound, min(raw_distance_reward, upBound))
        
        return bounded_reward

    
    #rewrite this so it uses near_wall instead
    def _heading_toward_wall_punish(self, modifier, margin=30):
        x, y = self.snake_pos
        reward = 0
        # Top wall
        if y < margin and self.direction == 0:
            reward -= modifier
        # Bottom wall
        if y > self.frame_size_y - margin - 10 and self.direction == 1:
            reward -= modifier
        # Left wall
        if x < margin and self.direction == 2:
            reward -= modifier
        # Right wall
        if x > self.frame_size_x - margin - 10 and self.direction == 3:
            reward -= modifier
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
            return -5
        else:
            # Reward for avoiding collision
            return 2
        
    def _distance_from_wall_reward(self):
        x, y = self.snake_pos
        dist_x = min(x, self.frame_size_x - x)
        dist_y = min(y, self.frame_size_y - y)
        min_dist = min(dist_x, dist_y)
        return min_dist / 100  # Reward staying near the center
    
    def _move_closer_reward(self, modifier):
        current_dist = self._get_food_distance()
        if current_dist < self.prev_food_dist:
            reward = modifier
        else:
            reward = 0
        self.prev_food_dist = current_dist
        return reward
    
    def _move_away_punish(self, modifier):
        current_dist = self._get_food_distance()
        if current_dist > self.prev_food_dist:
            penalty = -modifier
        else:
            penalty = 0
        self.prev_food_dist = current_dist
        return penalty


    def _survival(self, terminated):
        totalReward = 0
        totalReward += self._survival_reward(terminated, 0.1)
        totalReward += self._death_penalty(terminated, 50)
        totalReward += self._heading_toward_wall_punish(0.5)

        return totalReward

    def _length(self, terminated, ate_food, prev_direction):
        survive = self._survival_reward(terminated, 0.2)
        death_pen = self._death_penalty(terminated, 50)
        food_eaten = self._food_eaten_reward(ate_food, 50)
        move_closer = self._move_closer_reward(1)
        move_away = self._move_away_punish(0.5)
        #give reward for facing towards apple

        totalReward = survive + death_pen + food_eaten + move_closer + move_away

        if self.direction != prev_direction and self.direction == self._get_direction_to_food():
            totalReward += 2  # Reward for correct turn toward food
        if self.direction == prev_direction:
            self.straight_steps += 1
        else:
            self.straight_steps = 0
        if self.straight_steps > 10:
            totalReward -= 0.2  # Mild penalty for long straight sequences

        self.last_reward_breakdown = {
            "survival": survive,
            "death_penalty": death_pen,
            "food_eaten": food_eaten,
            "move_closer": move_closer,
            "move_away": move_away,
            "total": totalReward,
        }
        return totalReward