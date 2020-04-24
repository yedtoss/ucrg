"""
This module implements UCRG.
"""
import numpy as np
import itertools
import gambit

from . import utility


class UCRG:
    """
    Implements UCRG. At the moment only 2-players are supported.
    """

    def __init__(self, num_actions, player_id, delta=0.01):
        """

        :param num_actions: dict indicating the number of actions available for each player. Keys correspond to
                            player id and values to the number of actions available for the corresponding player.
                            NB: Only 2-players are supported currently
        :param player_id: The id of the current player.
        :param delta: The desired failing probability.
        """
        # In comments, unless otherwise states we used action to mean joint-action.

        self.player_id = player_id

        # Identify the id of the opponent as the other id in the dict different than player_id
        self.opponent_id = self.player_id
        for id_ in num_actions:
            if id_ != self.player_id:
                self.opponent_id = id_

        self.num_actions_with_id = num_actions

        # This is done for multiple reasons 1- It ensures synchronisation between the players, for example when
        # multiple actions are optimal. With this players will check actions in the same orders, thereby agreeing on
        # a single common action. 2- We expect to observe actions and their rewards ordered by the player'ids. So this
        # allows us to get it immediately.
        self.first_id, self.second_id = utility.order_ids((self.player_id, self.opponent_id))
        self.num_actions = (self.num_actions_with_id[self.first_id], self.num_actions_with_id[self.second_id])

        # Keep track of number of times action were taken from round 0 until current round
        self.num_taken_so_far = np.zeros(self.num_actions)
        # Keep track of number of times action were taken from round 0 until the final round of the previous episode.
        self.num_taken_until_last_episode = np.zeros(self.num_actions)
        # Keep track of number of times action were taken during the current artificial episode
        self.num_current_episode = np.zeros(self.num_actions)

        self.doubling_trick = utility.DoublingTrick()

        # To keep track of sum of rewards for each player'id
        self.sum_rewards = {self.player_id: np.zeros(self.num_actions),
                            self.opponent_id: np.zeros(self.num_actions)}

        self.time_step = 0.
        self.num_episodes = 0.
        self.num_steps_current_episode = 0.

        self.ACT_KEY = 'actions'
        self.PROB_KEY = 'prob'

        # The actions in policy are assumed ordered by player'ids
        self.policy = {self.ACT_KEY: [(0, 0)], self.PROB_KEY: [1]}

        self.MAX_R = 1.
        self.MIN_R = 0.
        self.delta = delta
        self.K = self.num_actions[0] * self.num_actions[1]  # Total number of joint-actions

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
        self.num_episodes = 0.
        self.num_steps_current_episode = 0.

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
            # Start of new episode
            self.num_episodes += 1

            confidence_probability = min(0.5, 1. / (self.num_episodes * self.time_step))
            confidence_log = 2 * np.log(1. / confidence_probability)

            delta_scaler = 6 * np.log2(max(2, self.time_step / (8. * self.K))) * self.K ** 2
            delta_scaler = 1 + 16 * np.log(self.time_step) * self.K + 2 * self.K
            confidence_error = self.delta / delta_scaler

            confidence_log = np.log(1. / confidence_error) / 1.99
            epsilon = np.cbrt(2. / 1.99) * np.cbrt(self.K * np.log(1. / confidence_error) / self.time_step)

            # When num_taken_so_far is 0 we set the confidence to 1.
            confidence_interval = np.sqrt(np.divide(confidence_log,
                                                    self.num_taken_so_far,
                                                    out=np.ones(self.num_actions),
                                                    where=self.num_taken_so_far > 0))

            rewards_up = {}  # To keep hold of optimistic rewards for each player
            rewards_down = {}  # To keep hold of pessimistic rewards for each player

            for id_, value_ in self.sum_rewards.items():
                rewards_up[id_] = np.minimum(self.MAX_R,
                                             np.divide(value_, self.num_taken_so_far, out=np.zeros(self.num_actions),
                                                       where=self.num_taken_so_far > 0) + confidence_interval)
                rewards_down[id_] = np.maximum(self.MIN_R,
                                               np.divide(value_, self.num_taken_so_far, out=np.zeros(self.num_actions),
                                                         where=self.num_taken_so_far > 0) - confidence_interval)

            #  Computing maximin policies/values
            maximin_best_response_policy_first, lower_maximin_value_first = self.compute_optimistic_maximin_policies(
                rewards_up[self.first_id], rewards_down[self.first_id], is_second=0)
            maximin_best_response_policy_second, lower_maximin_value_second = self.compute_optimistic_maximin_policies(
                rewards_up[self.second_id], rewards_down[self.second_id], is_second=1)

            # Getting advantage matrices
            advantage_matrix_first = rewards_up[self.first_id] - lower_maximin_value_first
            advantage_matrix_second = rewards_up[self.second_id] - lower_maximin_value_second
            advantage_matrix = {self.first_id: advantage_matrix_first, self.second_id: advantage_matrix_second}

            # Estimating the EBS policy
            ebs_advantage, ebs_policy = utility.EBS(self.num_actions_with_id,
                                                    self.first_id,
                                                    self.second_id).compute_from_advantage(advantage_matrix)
            self.policy = ebs_policy

            # Estimating the ideal action of each player
            approx_ideal_action_first, is_valid_first, ideal_advantage_first = self.find_approximate_ideal_action(
                advantage_matrix,
                epsilon,
                ebs_advantage,
                self.first_id, self.second_id)

            approx_ideal_action_second, is_valid_second, ideal_advantage_second = self.find_approximate_ideal_action(
                advantage_matrix,
                epsilon,
                ebs_advantage,
                self.second_id, self.first_id)

            ideal_valid_players = []
            approx_ideal_actions = []
            if is_valid_first:
                ideal_valid_players.append(ideal_advantage_first)
                approx_ideal_actions.append(approx_ideal_action_first)
            if is_valid_second:
                ideal_valid_players.append(ideal_advantage_second)
                approx_ideal_actions.append(approx_ideal_action_second)

            # If at the ideal action for one player, the other player is receiving larger than its ebs, then
            # we play this ideal action.
            if len(ideal_valid_players) > 0:
                approx_ideal_action = approx_ideal_actions[int(np.argmax(ideal_valid_players))]
                self.policy = {'actions': [approx_ideal_action], 'prob': [1.]}

            # If potential error on ebs policy is too large play the responsible action
            if 2. * self.compute_policy_interval(confidence_interval, ebs_policy) > epsilon:
                self.policy = {'actions': [self.find_conditional_argmax(confidence_interval, ebs_policy, epsilon)],
                               'prob': [1.]}

            # If potential error on the maximin policy on any player is too large play the responsible action
            # Note here importance of synchronizing using first_id and second_id consistently.
            if 2. * self.compute_policy_interval(confidence_interval, maximin_best_response_policy_first) > epsilon:
                self.policy = {'actions': [self.find_conditional_argmax(confidence_interval,
                                                                        maximin_best_response_policy_first, epsilon)],
                               'prob': [1.]}
            if 2. * self.compute_policy_interval(confidence_interval, maximin_best_response_policy_second) > epsilon:
                self.policy = {'actions': [self.find_conditional_argmax(confidence_interval,
                                                                        maximin_best_response_policy_second, epsilon)],
                               'prob': [1.]}

            self.num_current_episode = np.zeros(self.num_actions)
            self.num_taken_until_last_episode = self.num_taken_so_far.copy()
            self.num_steps_current_episode = 0.

    def act(self):
        """
        Performs an action in the environment
        :return: the action to play.
        """
        best_action = utility.find_action_to_play(self.policy, self.num_current_episode, self.num_steps_current_episode)
        action = best_action[0]
        if self.player_id == self.second_id:
            action = best_action[1]

        self.expected_previous_action = best_action

        return action

    def find_conditional_argmax_old(self, confidence_interval, policy_, epsilon):
        mask = np.nonzero(confidence_interval > epsilon / 2.)
        max_index = np.argmax(np.array(policy_['prob'])[mask])
        return tuple(np.array(mask)[:, max_index])

    def find_conditional_argmax(self, confidence_interval, policy_, epsilon):
        """
        Given that weighted policy confidence bonus exceeds a threshold (epsilon/2.), we want to find the action
        with the largest contribution (in term of probability) responsible for this exceed.
        :param confidence_interval: 2D numpy array indicating confidence bonus for each action
        :param policy_: The policy for which to compute the action with largest contribution
        :param epsilon: An indicator of the threshold.
        :return: The action with largest contribution
        """
        best_prob = -1
        best_action = (-1, -1)

        for i in range(len(policy_['prob'])):
            if 2. * confidence_interval[policy_['actions'][i]] > epsilon and policy_['prob'][i] > best_prob:
                best_prob = policy_['prob'][i]
                best_action = policy_['actions'][i]

        assert best_action != (-1, -1)
        return best_action

    def compute_policy_interval(self, confidence_interval, policy_):
        """

        :param confidence_interval: 2D numpy array indicating confidence bonus for each action
        :param policy_: The policy for which to compute its (weighted by probability) confidence bonus
        :return: The confidence bonus of the policy computed as weighted (by probability) average of actions played.
        """
        return min(1., np.dot(confidence_interval[tuple(np.array(policy_['actions']).T)], policy_['prob']))

    def find_approximate_ideal_action(self, advantage_matrix, epsilon, ebs_advantage, id_, not_id):
        """
        From the set of actions with positive advantage for id_, epsilon-larger than the EBS advantage of id_,
         find the one (best_action) maximizing not_id advantage. Then check if the advantage for best_action is
         larger than not_id ebs advantage.

        :param advantage_matrix: Dict containing the advantage game for each player. Key corresponds to player id
                                    and value to game matrix
        :param epsilon:
        :param ebs_advantage: dictionary containing the EBS advantage for each player. Key corresponds to player'id
                                and value are real number indicating the advantage value
        :param id_: The other player
        :param not_id:  The id of the the player for which the restricted (such that the other player advantage
                            is at least as large as its ebs) ideal action is being computed
        :return: A tuple of 3 values containing: 1- The ideal action for player with id `not_id`.
                    2- A bool which is True if the advantage of not_id when playing the ideal action is larger than
                    not_id ebs advantage. 3- A real indicating the advantage of not_id when playing the ideal
                    action computed.
        """

        all_actions = itertools.product(range(self.num_actions_with_id[self.first_id]),
                                        range(self.num_actions_with_id[self.second_id]))
        best_advantage = -1
        best_action = (-1, -1)

        for action in all_actions:
            if advantage_matrix[id_][action] >= 0 and advantage_matrix[id_][action] + epsilon >= ebs_advantage[id_]:
                # if advantage_matrix[id_][action] + epsilon >= ebs_advantage[id_]:
                if advantage_matrix[not_id][action] > best_advantage:
                    best_advantage = advantage_matrix[not_id][action]
                    best_action = action

        # This action should always exists since the ebs policy is a trivial solution?
        assert best_action != (-1, -1)

        # Return best_action and whether or not best_action advantage is larger than ebs_advantage
        return best_action, best_advantage > ebs_advantage[not_id], best_advantage

    def compute_optimistic_maximin_policies(self, reward_matrix_up, reward_matrix_down, is_second):
        """
        Compute an optimistic maximin policies
        :param reward_matrix_up: the optimistic rewards matrix
        :param reward_matrix_down:  the pessimistic rewards matrix
        :param is_second: whether or not we are looking for maximim of row or column player
        :return: tuple of two values: 1- the maximin policy and the maximin value
        """

        game_sign = (-1) ** is_second  # if the reward (identified by is_second) concern the row player (which is
        # always first_id), then the sign is 1 otherwise it is -1

        game_up_matrix = np.vectorize(gambit.Rational)(game_sign * reward_matrix_up)
        game_up = gambit.Game.from_arrays(game_up_matrix, -game_up_matrix)
        equilibria_profile = gambit.nash.lp_solve(game_up, rational=False)[0]
        maximin_policy = utility.get_numpy_strategy_from_gambit_profile(game_up, equilibria_profile, is_second)

        # Need to find the actions of the second player
        best_response_action_id = self.second_id
        if is_second:
            best_response_action_id = self.first_id

        # Focusing only on deterministic stationary policies
        attack_policy = np.zeros(self.num_actions_with_id[best_response_action_id])
        # best_response_policy = attack_policy.copy()
        best_response_action = 0
        # best_response_policy[best_response_action] = 1.

        game_down_matrix = np.vectorize(gambit.Rational)(game_sign * reward_matrix_down)
        game_down = gambit.Game.from_arrays(game_down_matrix, -game_down_matrix)

        # Create a strategy profile for game_down with the maximin policy set to the correct player
        game_strategy_profile = utility.create_gambit_strategy_from_array(game_down, is_second, maximin_policy)

        lower_maximin_value = self.MAX_R  # could also take the max value in reward_matrix_down

        for pos in range(len(attack_policy)):
            attack_policy[pos] = 1.

            utility.set_gambit_strategy_from_array(game_down, game_strategy_profile, int(not is_second), attack_policy)
            current_value = game_strategy_profile.payoff(game_down.players[is_second])

            if current_value < lower_maximin_value:
                lower_maximin_value = current_value
                # best_response_policy = attack_policy.copy()
                best_response_action = pos

            attack_policy[pos] = 0.

        maximin_best_response_actions = [
            (best_response_action, single_action) if is_second else (single_action, best_response_action)
            for single_action in range(len(maximin_policy)) if maximin_policy[single_action] > 0]

        maximin_best_response_probs = [
            maximin_policy[action[1]] if is_second else maximin_policy[action[0]]
            for action in maximin_best_response_actions]

        maximin_best_response_policy = {'actions': maximin_best_response_actions,
                                        'prob': maximin_best_response_probs}

        return maximin_best_response_policy, lower_maximin_value

    def name(self):
        """
        :return: the name of this algorithm
        """
        return 'UCRG'
