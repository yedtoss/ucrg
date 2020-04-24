"""
This module simulates an experiment
"""
import numpy as np
import runstats


class Simulator:
    def __init__(self, environment, algorithms_agents, horizon):
        """

        :param environment: the environment
        :param algorithms_agents: list of agents to test
        :param horizon: int, the horizon to run the simulation for
        """
        self.environment = environment
        self.algorithms_agents = algorithms_agents
        self.num_agents = len(self.algorithms_agents[0])
        self.num_algorithms = len(self.algorithms_agents)
        self.horizon = horizon

        self.cumulative_rewards = [
            [[runstats.Statistics() for _ in range(self.horizon)] for _ in range(self.num_agents)] for _ in
            range(self.num_algorithms)]

    def run(self, num_trials=1):
        """
        Run the simulation
        :param num_trials: number of independent trials
        :return:
        """

        for trial in range(num_trials):
            self.environment.reset()

            for algorithm_id in range(self.num_algorithms):
                self.environment.reset_same_trial_new_algorithm()
                for agent in self.algorithms_agents[algorithm_id]:
                    agent.reset()
                current_cumul_rewards = [np.zeros(self.horizon) for _ in range(self.num_agents)]
                for time_step in range(self.horizon):

                    env_action = tuple(agent.act() for agent in self.algorithms_agents[algorithm_id])
                    env_rewards = self.environment.act(env_action)

                    action = {self.algorithms_agents[algorithm_id][i].player_id: env_action[i] for i in
                              range(self.num_agents)}
                    rewards = {self.algorithms_agents[algorithm_id][i].player_id: env_rewards[i] for i in
                               range(self.num_agents)}
                    for agent in self.algorithms_agents[algorithm_id]:
                        agent.observe(action, rewards)

                    for i in range(self.num_agents):
                        current_cumul_rewards[i][time_step] = env_rewards[i]

                for i in range(self.num_agents):
                    current_cumul_rewards[i] = np.cumsum(current_cumul_rewards[i])

                for i in range(self.num_agents):
                    for time_step in range(self.horizon):
                        self.cumulative_rewards[algorithm_id][i][time_step].push(current_cumul_rewards[i][time_step])

        return self

    def print_statistics(self):
        for algorithm_id in range(self.num_algorithms):
            print('Algorithm ', self.algorithms_agents[algorithm_id][0].name())
            print('Final rewards', tuple(cumul[-1].mean() for cumul in self.cumulative_rewards[algorithm_id]))
            print('Final mean rewards',
                  tuple(cumul[-1].mean() / len(cumul) for cumul in self.cumulative_rewards[algorithm_id]))
            print('Final std dev', tuple(cumul[-1].stddev() for cumul in self.cumulative_rewards[algorithm_id]))
            print('')
