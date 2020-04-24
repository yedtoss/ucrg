"""
This module implements explore then commit for solving EBS
"""
from OptimisticEBS import utility
import numpy as np
import itertools
import gambit


class ExploreThenCommit:

    def __init__(self, num_actions, player_id, delta=0.01):
        """

        :param num_actions: dict indicating the number of actions available for each player. Keys correspond to
                            player id and values to the number of actions available for the corresponding player.
                            NB: Only 2-players are supported currently
        :param player_id: The id of the current player.
        :param delta: The desired failing probability.
        """

        self.player_id = player_id
        self.opponent_id = self.player_id
        for id_ in num_actions:
            if id_ != self.player_id:
                self.opponent_id = id_

        self.num_actions_with_id = num_actions

        # This is done for multiple reasons 1- It ensures synchronisation between the players, for example when
        # multiple actions are optimal. With this players will check actions in the same orders, thereby agreeing on
        # a single common action. 2- We expect to observe actions and their rewards ordered by the player_id. So this
        # allows us to get it immediately.
        self.first_id, self.second_id = utility.order_ids((self.player_id, self.opponent_id))
        self.num_actions = (self.num_actions_with_id[self.first_id], self.num_actions_with_id[self.second_id])

        self.num_taken_so_far = np.zeros(self.num_actions)
        self.num_taken_until_last_episode = np.zeros(self.num_actions)
        self.num_current_episode = np.zeros(self.num_actions)

        self.doubling_trick = utility.DoublingTrick()

        self.sum_rewards = {self.player_id: np.zeros(self.num_actions),
                            self.opponent_id: np.zeros(self.num_actions)}

        self.time_step = 0.
        self.num_steps_current_episode = 0.
        self.num_episodes = 0

        self.all_actions = list(itertools.product(range(self.num_actions_with_id[self.first_id]),
                                                  range(self.num_actions_with_id[self.second_id])))

        self.exploration_phase = True
        self.exploration_action_id = 0
        self.exploration_m = 1

        self.K = self.num_actions[0] * self.num_actions[1]

        self.ACT_KEY = 'actions'
        self.PROB_KEY = 'prob'

        self.policy = {self.ACT_KEY: [(0, 0)], self.PROB_KEY: [1]}

        self.MAX_R = 1

        self.delta = delta

        # This contain the action pair that was expected to be played
        self.expected_previous_action = (0, 0)

    def reset(self):
        """
        Reset the current instance such that it can be used as if it hasn't observed any rewards/played any actions.
        :return:
        """
        self.num_taken_so_far = np.zeros(self.num_actions)
        self.num_taken_until_last_episode = np.zeros(self.num_actions)
        self.num_current_episode = np.zeros(self.num_actions)

        self.doubling_trick = utility.DoublingTrick()

        self.sum_rewards = {self.player_id: np.zeros(self.num_actions),
                            self.opponent_id: np.zeros(self.num_actions)}

        self.time_step = 0.
        self.num_steps_current_episode = 0.
        self.num_episodes = 0

        self.exploration_phase = True
        self.exploration_action_id = 0
        self.exploration_m = 1

        self.policy = {self.ACT_KEY: [(0, 0)], self.PROB_KEY: [1]}

        self.expected_previous_action = (0, 0)

    def observe(self, action, reward):
        """
        Observe the rewards for the played action
        :param action: List of actions played ordered by player_ids
        :param reward: List of rewards obtained ordered by player_ids
        :return:
        """
        action = (action[self.first_id], action[self.second_id])

        # Checking that opponent actually played its expected action
        opponent_action = action[1] if self.opponent_id == self.second_id else action[0]
        expected_opponent_action = (self.expected_previous_action[1] if self.opponent_id == self.second_id else
                                    self.expected_previous_action[0])
        assert opponent_action == expected_opponent_action

        self.num_taken_so_far[action] += 1
        self.num_current_episode[action] += 1
        self.sum_rewards[self.first_id][action] += reward[self.first_id]
        self.sum_rewards[self.second_id][action] += reward[self.second_id]

        self.time_step += 1
        self.num_steps_current_episode += 1

        if self.doubling_trick(self.num_taken_until_last_episode, self.num_current_episode, action):
            self.num_episodes += 1
            self.exploration_phase = True
            self.exploration_action_id = 0
            confidence_probability = min(0.5, 1. / (self.num_episodes * self.time_step))

            delta_scaler = 6 * np.log2(max(2, self.time_step/(8.*self.K))) * self.K**2
            delta_scaler = 1 + 16 * np.log(self.time_step) * self.K + 2 * self.K
            confidence_probability = self.delta/min(1, delta_scaler)

            self.exploration_m = np.ceil(
                ((2 * self.time_step) ** (2. / 3)) * (self.K * np.log(self.K / confidence_probability) ** (1. / 3)))

            rewards = {}

            for id_, value_ in self.sum_rewards.items():
                rewards[id_] = np.minimum(self.MAX_R,
                                          np.divide(value_, self.num_taken_so_far, out=np.zeros(self.num_actions),
                                                    where=self.num_taken_so_far > 0))

            game_first_matrix = np.vectorize(gambit.Rational)(rewards[self.first_id])
            game_first = gambit.Game.from_arrays(game_first_matrix, -game_first_matrix)

            game_second_matrix = np.vectorize(gambit.Rational)(-rewards[self.second_id])
            game_second = gambit.Game.from_arrays(game_second_matrix, -game_second_matrix)

            advantage_matrix = {
                self.first_id: rewards[self.first_id] - gambit.nash.lp_solve(game_first, rational=False)[0].payoff(
                    game_first.players[0]),
                self.second_id: rewards[self.second_id] - gambit.nash.lp_solve(game_second, rational=False)[
                    0].payoff(game_second.players[1])
            }

            _, ebs_policy = utility.EBS(self.num_actions_with_id,
                                        self.first_id,
                                        self.second_id).compute_from_advantage(advantage_matrix)

            self.policy = ebs_policy

            self.num_current_episode = np.zeros(self.num_actions)
            self.num_taken_until_last_episode = self.num_taken_so_far.copy()
            self.num_steps_current_episode = 0.

    def act(self):
        """
        Performs an action in the environment
        :return: the action to play.
        """
        if self.exploration_phase:
            all_actions_id = range(self.exploration_action_id, self.K)
            for action_id in all_actions_id:
                self.exploration_action_id = action_id
                if self.num_taken_so_far[self.all_actions[action_id]] < self.exploration_m:
                    self.expected_previous_action = self.all_actions[action_id]
                    action_to_play = self.all_actions[action_id][0]
                    if self.player_id == self.second_id:
                        action_to_play = self.all_actions[action_id][1]
                    return action_to_play

        self.exploration_phase = False

        best_action = utility.find_action_to_play(self.policy, self.num_current_episode, self.num_steps_current_episode)
        action = best_action[0]
        if self.player_id == self.second_id:
            action = best_action[1]

        self.expected_previous_action = best_action

        return action

    def name(self):
        """
        :return: the name of this algorithm
        """
        return 'ETC'
