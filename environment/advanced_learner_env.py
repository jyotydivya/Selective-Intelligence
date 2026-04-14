import numpy as np
import random


class AdvancedLearnerEnv:

    def __init__(self, learner_type=None):

        # 0 = Beginner, 1 = Intermediate, 2 = Advanced
        if learner_type is None:
            self.learner_type = np.random.choice([0, 1, 2])
        else:
            self.learner_type = learner_type

        # ----------------------------------------
        # INITIAL STATES (DIFFERENT START POINTS)
        # ----------------------------------------
        if self.learner_type == 0:  # Beginner
            self.skill_level = 0
            self.performance = 0
            self.frustration = 2
            self.help_dependency = 2
            self.difficulty = 0

        elif self.learner_type == 1:  # Intermediate
            self.skill_level = 1
            self.performance = 1
            self.frustration = 1
            self.help_dependency = 1
            self.difficulty = 1

        else:  # Advanced
            self.skill_level = 2
            self.performance = 2
            self.frustration = 0
            self.help_dependency = 0
            self.difficulty = 2

        self.recent_correct = 0


    # ----------------------------------------
    # STATE
    # ----------------------------------------
    def get_state(self):
        return (
            self.learner_type,
            self.skill_level,
            self.performance,
            self.frustration,
            self.help_dependency,
            self.difficulty
        )


    # ----------------------------------------
    # SUCCESS PROBABILITY
    # ----------------------------------------
    def compute_success_probability(self, action):

        gap = self.skill_level - self.difficulty
        base_prob = 0.45 + 0.15 * gap

        frustration_effect = {0: +0.10, 1: 0.0, 2: -0.15}
        performance_effect = {0: -0.10, 1: 0.0, 2: +0.10}
        help_effect = {0: +0.05, 1: 0.0, 2: -0.10}

        # 🔥 STRONG learner difference
        learner_bias = {
            0: -0.20,   # Beginner struggles
            1: 0.0,
            2: +0.20    # Advanced strong
        }

        # 🔥 Action preference differs per learner
        if self.learner_type == 0:
            action_effect = [0.0, 0.15, 0.08, 0.35]  # likes explanation
        elif self.learner_type == 1:
            action_effect = [0.0, 0.10, 0.10, 0.20]
        else:
            action_effect = [0.10, 0.05, 0.15, 0.05]  # prefers autonomy

        prob = base_prob
        prob += learner_bias[self.learner_type]
        prob += frustration_effect[self.frustration]
        prob += performance_effect[self.performance]
        prob += help_effect[self.help_dependency]
        prob += action_effect[action]

        return np.clip(prob, 0.05, 0.95)


    # ----------------------------------------
    # STEP FUNCTION
    # ----------------------------------------
    def step(self, action):

        prob = self.compute_success_probability(action)
        success = np.random.rand() < prob

        # ----------------------------------------
        # 🔥 STRONG REWARD DIFFERENTIATION
        # ----------------------------------------
        if self.learner_type == 0:  # Beginner

            if success:
                reward = {
                    0: 5,
                    1: 15,
                    2: 10,
                    3: 20
                }[action]
            else:
                reward = -10

        elif self.learner_type == 1:  # Intermediate

            if success:
                reward = {
                    0: 10,
                    1: 12,
                    2: 12,
                    3: 8
                }[action]
            else:
                reward = -8

        else:  # Advanced

            if success:
                reward = {
                    0: 20,
                    1: 8,
                    2: 15,
                    3: 2
                }[action]
            else:
                reward = -6


        # ----------------------------------------
        # SKILL GROWTH (DIFFERENT SPEEDS)
        # ----------------------------------------
        growth_rate = {
            0: 0.20,
            1: 0.30,
            2: 0.50
        }

        if success and random.random() < growth_rate[self.learner_type]:
            self.skill_level = min(3, self.skill_level + 1)


        # ----------------------------------------
        # PERFORMANCE UPDATE
        # ----------------------------------------
        if success:
            self.recent_correct += 1
        else:
            self.recent_correct -= 1

        self.recent_correct = np.clip(self.recent_correct, -3, 3)

        if self.recent_correct >= 2:
            self.performance = min(2, self.performance + 1)

        if self.recent_correct <= -2:
            self.performance = max(0, self.performance - 1)


        # ----------------------------------------
        # FRUSTRATION (VERY DIFFERENT)
        # ----------------------------------------
        frustration_rate = {
            0: 0.9,
            1: 0.5,
            2: 0.2
        }

        if success:
            self.frustration = max(0, self.frustration - 1)
        else:
            if random.random() < frustration_rate[self.learner_type]:
                self.frustration = min(2, self.frustration + 1)


        # ----------------------------------------
        # HELP DEPENDENCY
        # ----------------------------------------
        if action == 3:
            self.help_dependency = min(2, self.help_dependency + 1)
        else:
            if success:
                self.help_dependency = max(0, self.help_dependency - 1)


        # ----------------------------------------
        # DIFFICULTY
        # ----------------------------------------
        if success and random.random() < 0.25:
            self.difficulty = min(3, self.difficulty + 1)

        if not success and random.random() < 0.20:
            self.difficulty = max(0, self.difficulty - 1)


        # ----------------------------------------
        # 🔥 FINAL FIX: SCALE REWARD BY LEARNER TYPE
        # ----------------------------------------
        reward_scale = {
            0: 0.7,   # Beginner → lower rewards
            1: 1.0,   # Intermediate
            2: 1.5    # Advanced → higher rewards
        }

        reward *= reward_scale[self.learner_type]

        return self.get_state(), reward, success