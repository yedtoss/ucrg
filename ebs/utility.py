"""
This module contains utility functions.
"""
import numpy as np
import itertools


def get_numpy_strategy_from_gambit_profile(game, mixed_profile, player_id):
    """
    Given a mixed strategy profile in gambit format, convert it to numpy array for a specific player
    :param game: the gambit game
    :param mixed_profile: the gambit mixed strategy profile
    :param player_id: the player id (int) for which one is converting its mixed strategy profile
    :return: A numpy array representing the mixed strategy profile of the player.
    """
    return np.array([mixed_profile[strategy] for strategy in game.players[player_id].strategies])


def set_gambit_strategy_from_array(game, mixed_profile, player_id, probs):
    """
    Set a mixed gambit strategy profile from a numpy array for a specific player
    :param game: the gambit game
    :param mixed_profile: the gambit mixed strategy profile to set
    :param player_id: the id (int) of the player
    :param probs: the numpy array containing probability of action for the specified player
    :return:
    """
    for i in range(len(probs)):
        mixed_profile[game.players[player_id].strategies[i]] = probs[i]


def create_gambit_strategy_from_array(game, player_id, probs):
    """
    Return a mixed gambit strategy profile from a numpy array for a specific player
    :param game: the gambit game
    :param player_id: the id (int) of the player
    :param probs: the numpy array containing probability of action for the specified player
    :return: the gambit mixed strategy profile
    """
    mixed_profile = game.mixed_strategy_profile()
    set_gambit_strategy_from_array(game, mixed_profile, player_id, probs)
    return mixed_profile


def order_ids(ids):
    """
    This returned the ids sorted
    :param ids: array, tuple, iterator. The list of ids
    :return: A sorted tuple of the ids
    """
    return tuple(set(ids))


def find_action_to_play(policy, num_current_episode, num_steps_current_episode):
    """
    This function return the action to play such that the empirical probability of play is as close as possible
    to the target policy.
    :param policy: The target policy. A dict with 2 keys: 'actions' An array of tuple containing the
                    actions played by this policy. 'probs': An array of float (same size as 'actions') indicating
                    the probability of playing the corresponding action
    :param num_current_episode: A 2D numpy array indicating the number of times the corresponding action has been
                                played so far
    :param num_steps_current_episode: float, indicating the total number of play so far
    :return: the action to play such that the empirical probability of play is as close as possible
                to the target policy.
    """
    best_action = policy['actions'][0]
    best_loss = 2

    for id_ in range(len(policy['actions'])):
        current_action = policy['actions'][id_]
        current_loss = abs(policy['prob'][id_] -
                           (num_current_episode[current_action] + 1.) /
                           (num_steps_current_episode + 1.)
                           )

        for id2 in range(len(policy['actions'])):
            if id2 != id_:
                current_action2 = policy['actions'][id2]
                current_loss = max(current_loss, abs(
                    policy['prob'][id2] - (num_current_episode[current_action2]) / (
                            num_steps_current_episode + 1.)))

        if current_loss < best_loss:
            best_loss = current_loss
            best_action = current_action

    return best_action


class DoublingTrick:
    """
    This implements the doubling trick
    """

    def __init__(self):
        pass

    def __call__(self, num_taken_until_last_episode, num_current_episode, action):
        """
        This method return True when a new episode should be started according to the doubling trick criterion.
        :param num_taken_until_last_episode: 2D numpy array indicating how many times each action had been played
                                                from first round until the final round the last completed episode
        :param num_current_episode: 2D numpy array indicating how many times each action has been played
                                    during the current episode
        :param action: A tuple of two int, indicating the action that has just been played
        :return: True if a new episode should be started, False otherwise.
        """
        return num_current_episode[action] > max(1, num_taken_until_last_episode[action])


class EBS:
    """
    This computes the EBS for a game with known expected payoffs.
    """

    def __init__(self, num_actions_with_id, first_id, second_id):
        """

        :param num_actions_with_id: dict containing the number of actions available for each player id
        :param first_id: the id of the first player
        :param second_id: the id of the second player
        """
        self.num_actions_with_id = num_actions_with_id
        self.first_id = first_id
        self.second_id = second_id

    def compute_from_advantage(self, advantage_matrix):
        """
        Compute the ebs from an advantage matrix
        :param advantage_matrix: dict of 2D numpy array indicating the advantage matrix for each player id
        :return: A tuple of two values. 1- the EBS advantage 2- the EBS policy
        """

        advantage_matrix_first = advantage_matrix[self.first_id]
        advantage_matrix_second = advantage_matrix[self.second_id]

        num_actions = (self.num_actions_with_id[self.first_id], self.num_actions_with_id[self.second_id])
        all_pairs_of_actions = itertools.product(range(num_actions[0]), range(num_actions[1]),
                                                 range(num_actions[0]), range(num_actions[1]))

        best_policy = {'actions': [(0, 0), (0, 0)],
                       'prob': [1, 0]}
        best_full_score = {self.first_id: advantage_matrix_first[best_policy['actions'][0]],
                           self.second_id: advantage_matrix_second[best_policy['actions'][0]]}
        best_score = min(best_full_score.values())

        for policy in all_pairs_of_actions:

            a0, a1, a2, a3 = policy[0], policy[1], policy[2], policy[3]

            condition1 = (advantage_matrix_first[(a0, a1)] <= advantage_matrix_second[(a0, a1)]
                          and advantage_matrix_first[(a2, a3)] <= advantage_matrix_second[(a2, a3)])

            condition2 = (advantage_matrix_first[(a0, a1)] >= advantage_matrix_second[(a0, a1)]
                          and advantage_matrix_first[(a2, a3)] >= advantage_matrix_second[(a2, a3)])

            if condition1:
                w = 0.
            elif condition2:
                w = 1.
            else:
                w = ((advantage_matrix_second[(a2, a3)] - advantage_matrix_first[(a2, a3)]) /
                     ((advantage_matrix_first[(a0, a1)] - advantage_matrix_first[(a2, a3)]) +
                      (advantage_matrix_second[(a2, a3)] - advantage_matrix_second[(a0, a1)])))

            full_score = {self.first_id:
                              w * advantage_matrix_first[(a0, a1)] + (1 - w) * advantage_matrix_first[(a2, a3)],
                          self.second_id:
                              w * advantage_matrix_second[(a0, a1)] + (1 - w) * advantage_matrix_second[(a2, a3)]}

            score = min(full_score.values())

            if score > best_score:
                best_score = score
                best_full_score = full_score
                best_policy = {'actions': [(a0, a1), (a2, a3)], 'prob': [w, 1 - w]}

        return best_full_score, best_policy
