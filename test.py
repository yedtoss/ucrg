"""
This run all the experiments
"""
from OptimisticEBS import ucrg
from OptimisticEBS import environment
from OptimisticEBS import simulator
from OptimisticEBS import utility
from OptimisticEBS import explore_then_commit
from OptimisticEBS import plot

import numpy as np
import gambit

if __name__ == "__main__":
    horizon = 100000
    num_trials = 50
    num_trials = 5

    # Example
    matrix_a_example = np.array([[4. / 5, 1. / 10], [9. / 5, 3. / 10]]) / 2.
    matrix_b_example = np.array([[4. / 5, 9. / 5], [0., 3. / 10]]) / 2.

    # Lower bound
    epsilon = 4 ** (1. / 3) * (horizon ** (-1. / 3))
    epsilon = 0
    matrix_a_lb = np.array([[0.5, 0.5], [0.5 + epsilon, 0.5]])
    matrix_b_lb = np.array([[1, 0.5], [0.5 + epsilon, 0.5]])

    # Generalized rock-paper-scissor Chaos in learning a simple two-person game  https://www.pnas.org/content/99/7/4748
    epsilon_a = 0.5
    epsilon_b = 0.5
    matrix_a_rpc = (np.array([[epsilon_a, -1, 1], [1, epsilon_a, -1], [-1, 1, epsilon_a]]) + 2.) / 4.
    matrix_b_rpc = (np.array([[epsilon_b, -1, 1], [1, epsilon_b, -1], [-1, 1, epsilon_b]]) + 2.) / 4.

    environments_data = [
        {'name': 'example', 'matrices': (matrix_a_example, matrix_b_example)},
        {'name': 'lower_bound', 'matrices': (matrix_a_lb, matrix_b_lb)},
        {'name': 'rpc', 'matrices': (matrix_a_rpc, matrix_b_rpc)},
    ]

    for data in environments_data:

        env_name = data['name']
        print('Processing Env ', env_name)
        matrix_a, matrix_b = data['matrices']

        num_K = matrix_a.shape[0] * matrix_a.shape[1]

        # test_env = environment.MatrixEnvironment(matrix_a, matrix_b)
        test_env = environment.FastBernoulliMatrixEnvironment(matrix_a, matrix_b, horizon)

        agent1_id = 0
        agent2_id = 1

        delta = 0.01

        num_actions = {0: matrix_a.shape[0], 1: matrix_a.shape[1]}
        agent1 = ucrg.UCRG(num_actions=num_actions, player_id=agent1_id, delta=delta)
        agent2 = ucrg.UCRG(num_actions=num_actions, player_id=agent2_id, delta=delta)
        ebs_agents = [agent1, agent2]

        explore_agent1 = explore_then_commit.ExploreThenCommit(num_actions=num_actions, player_id=agent1_id,
                                                               delta=delta)
        explore_agent2 = explore_then_commit.ExploreThenCommit(num_actions=num_actions, player_id=agent2_id,
                                                               delta=delta)
        explore_agents = [explore_agent1, explore_agent2]

        algorithms_agents = [ebs_agents, explore_agents]

        simulations = simulator.Simulator(environment=test_env, algorithms_agents=algorithms_agents,
                                          horizon=horizon).run(num_trials=num_trials)
        simulations.print_statistics()
        simulations_names = [algorithm[0].name() for algorithm in algorithms_agents]

        matrix_a_gambit = np.vectorize(gambit.Rational)(matrix_a)
        matrix_b_gambit = np.vectorize(gambit.Rational)(-matrix_b)
        maximin1 = gambit.nash.lp_solve(gambit.Game.from_arrays(matrix_a_gambit, -matrix_a_gambit), rational=False)[
            0].payoff(0)
        maximin2 = gambit.nash.lp_solve(gambit.Game.from_arrays(matrix_b_gambit, -matrix_b_gambit), rational=False)[
            0].payoff(1)

        advantage1 = matrix_a - maximin1
        advantage2 = matrix_b - maximin2

        advantage_matrix = {agent1_id: advantage1, agent2_id: advantage2}

        ebs_value, ebs_policy = utility.EBS(num_actions_with_id=num_actions, first_id=agent1_id,
                                            second_id=agent2_id).compute_from_advantage(
            advantage_matrix=advantage_matrix)

        print('True Maximin', (maximin1, maximin2))
        print('True EBS', ebs_value)
        print('True EBS policy ', ebs_policy)

        optimal_rewards = [np.cumsum((ebs_value[0] + maximin1) * np.ones(horizon)),
                           np.cumsum((ebs_value[1] + maximin2) * np.ones(horizon))]
        algo_regrets = [None] * len(algorithms_agents)
        algo_regrets_index = [None] * len(algorithms_agents)
        algo_names = simulations_names.copy()

        for i in range(len(algorithms_agents)):
            regrets = [optimal_rewards[0] - np.array([stats.mean() for stats in simulations.cumulative_rewards[i][0]]),
                       optimal_rewards[1] - np.array([stats.mean() for stats in simulations.cumulative_rewards[i][1]])]
            algo_regrets[i] = np.maximum(regrets[0], regrets[1])
            algo_regrets_index[i] = (regrets[0] < regrets[1]).astype(np.int)

        lb_regret = np.array([(num_K ** (1. / 3)) * (t ** (2. / 3)) / 4. for t in range(1, horizon + 1)])
        ub_regret = np.array(
            [4.007 * ((num_K * np.log((16 * num_K * np.log(t) + 2 * num_K + 1) / delta)) ** (1. / 3)) * (t ** (2. / 3))
             for t in
             range(1, horizon + 1)])

        algo_regrets.append(ub_regret)
        algo_regrets.append(lb_regret)
        algo_names.append('UB')
        algo_names.append('LB')

        deviations = [np.zeros(horizon) for _ in range(len(algo_regrets))]

        for i in range(len(algorithms_agents)):
            deviations[i] = np.array(
                [simulations.cumulative_rewards[i][algo_regrets_index[i][j]][j].stddev() for j in range(horizon)])

        plot.plot_regret(algo_regrets, algo_names, deviations, env_name)
