"""
This module contains 2-player multi armed bandits environments
"""
from scipy import stats
import numpy as np
import itertools


class MatrixEnvironment:
    """
    Implements a simple 2 players matrix game
    """

    def __init__(self, game_matrix_a, game_matrix_b):
        """
        :param game_matrix_a: 2D numpy array, matrix payoff for the first player
        :param game_matrix_b: 2D numpy array, matrix payoff for the second player
        """
        self.game_matrix_a = game_matrix_a
        self.game_matrix_b = game_matrix_b

    def act(self, action):
        """
        Return the reward for each player upon playing action.
        :param action: the action played
        :return: tuple, the reward of each player
        """
        return self.game_matrix_a[action], self.game_matrix_b[action]

    def reset(self):
        """
        Reset the environment such that it is as newly initialized
        :return:
        """
        pass

    def reset_same_trial_new_algorithm(self):
        """
        Reset the environment such that a new algorithm in the same trial can be executed.
        :return:
        """
        pass


class FastBernoulliMatrixEnvironment:
    """
    Implements a 2-player multi-armed bandit with bernoulli rewards.
    This class pre-generate all possible rewards and make sure that in the same trial, different algorithms would
    observe the same rewards upon playing some actions the same number of times
    """
    def __init__(self, game_matrix_a, game_matrix_b, horizon):
        """
        :param game_matrix_a: 2D numpy array, matrix payoff for the first player indicating the expectation of the
                                Bernoulli distribution
        :param game_matrix_b: 2D numpy array, matrix payoff for the second player indicating the expectation of the
                                Bernoulli distribution
        :param horizon: The target horizon.
        """
        self.game_matrix_a = game_matrix_a
        self.game_matrix_b = game_matrix_b
        self.game_matrices = [self.game_matrix_a, self.game_matrix_b]
        self.horizon = horizon

        self.full_K = self.game_matrix_a.shape
        self.all_actions = itertools.product(range(self.full_K[0]), range(self.full_K[1]))

        self.num_actions_so_far = np.zeros(self.full_K).astype(np.int)

        actions_dict = {action: [] for action in self.all_actions}
        self.rewards = [actions_dict.copy(), actions_dict.copy()]

        self.reset()

    def act(self, action):
        """
        Return the reward for each player upon playing action.
        :param action: the action played
        :return: tuple, the reward of each player
        """

        rew = self.rewards[0][action][self.num_actions_so_far[action]], self.rewards[1][action][
            self.num_actions_so_far[action]]

        self.num_actions_so_far[action] += 1
        return rew

    def reset(self):
        """
        Reset the environment such that it is as newly initialized.
        It pre-generate all possibles rewards from the Bernoulli distributions.
        :return:
        """
        self.reset_same_trial_new_algorithm()
        for i in range(2):
            for key in self.rewards[i]:
                self.rewards[i][key] = stats.bernoulli(self.game_matrices[i][key]).rvs(size=self.horizon)

    def reset_same_trial_new_algorithm(self):
        """
        Reset the environment such that a new algorithm in the same trial can be executed.
        It makes sure that in the same trial, different algorithms would
        observe the same rewards upon playing some actions the same number of times by not resetting the generated
        rewards
        :return:
        """
        self.num_actions_so_far = np.zeros(self.full_K).astype(np.int)


class BernoulliMatrixEnvironment:
    """
    Implements a 2-player multi-armed bandit with bernoulli rewards.
    """
    def __init__(self, game_matrix_a, game_matrix_b):
        self.game_matrix_a = game_matrix_a
        self.game_matrix_b = game_matrix_b

    def act(self, action):
        """
        Return the reward for each player upon playing action.
        :param action: the action played
        :return: tuple, the reward of each player
        """
        return stats.bernoulli(self.game_matrix_a[action]).rvs(), stats.bernoulli(self.game_matrix_b[action]).rvs()

    def reset(self):
        """
        Reset the environment such that it is as newly initialized.
        :return:
        """
        pass

    def reset_same_trial_new_algorithm(self):
        """
        Reset the environment such that a new algorithm in the same trial can be executed.
        :return:
        """
        pass
