    def _length(self, terminated, ate_food, prev_direction):
        totalReward = 0
        totalReward += self._survival_reward(0.001, terminated)
        totalReward += self._death_penalty(terminated, 50)
        #totalReward += self._heading_toward_wall_punish(0.5)
        totalReward += self._food_eaten_reward(ate_food, 50)
        #totalReward += self._wall_evasion_reward(prev_direction, 10)
        totalReward += self._food_distance_based_reward(1,-1)
        #totalReward += self._turning_to_food_reward(prev_direction,10)
        #totalReward += self._axis_direction_reward(2)

        

        return totalReward


#survival to 0, tuning death penalty to 100, upper bound to 0.5, makes snake just go in circles more tightly


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this one trained for a million, moves towards food but past it and self-terminates
    def _length(self, terminated, ate_food, prev_direction):
        totalReward = 0
        totalReward += self._survival_reward(0, terminated)
        totalReward += self._death_penalty(terminated, 50)
        totalReward += self._food_eaten_reward(ate_food, 50)
        #totalReward += self._food_distance_based_reward(0.5,-1)
        totalReward += self.food_approach_reward(3)


        return totalReward


# trained for 500000, goes towards wall and continuously circles
    def _length(self, terminated, ate_food, prev_direction):
        totalReward = 0
        totalReward += self._survival_reward(0, terminated)
        totalReward += self._death_penalty(terminated, 30)
        totalReward += self._food_eaten_reward(ate_food, 100)
        #totalReward += self._food_distance_based_reward(0.5,-1)
        totalReward += self._food_approach_reward(1)
        totalReward += self._wall_evasion_reward(prev_direction, 5)


        return totalReward


# trained for 300000, ent_coef 0.05, goes straight into wall
def length():
        totalReward = 0
        totalReward += self._death_penalty(terminated, 10)
        totalReward += self._food_eaten_reward(ate_food, 100)
        totalReward += self._move_closer_reward(2)
        totalReward += self._move_away_punish(3)

