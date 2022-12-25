"""自定义环境"""

import gym
import numpy as np
import tqdm
from gym import spaces

from Zh_Long.LyapunovComplex import get_step
from Zh_Long.delay_analysis_QueueAndSNCForLag import get_dcp_snc

np.random.seed(1)
rewards = []
actions = []
Pks = []
Uks = []
targets = []


class Agent:
    def __init__(self, i, loops):
        self.No = i
        self.Pk = 0.0
        self.U = 0.0
        self.B = 0.0
        self.p = 0.0
        self.lamda = 0.0
        self.snr = 0.0
        self.base = 0.0
        self.reward = 0.0
        self.Ps = []
        self.a_s = []
        self.loops = loops

    def getReward(self, snr) -> float:
        self.Pk, self.U, target, self.B = get_step(self.lamda, snr, self.Ps, self.a_s, loops=self.loops)
        return target


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, num_agents, num_states, p_min, p_max, p_sumMax, maxStep, loops, trainStart, model):
        super(CustomEnv, self).__init__()
        self.p_max = p_max
        self.p_min = p_min
        self.B_min = 0.0
        self.B_max = 0.0
        self.mius = [1 for _ in range(num_agents)]
        self.num_states = num_states
        self.p_sumMax = p_sumMax
        self.num_agents = num_agents
        self.agents = [Agent(i, loops) for i in range(self.num_agents)]
        self.steps = 0
        self.episodes = 0
        self.maxStep = maxStep
        self.base = 0.0
        self.reward = 0.0
        self.p_sum = 0.0
        self.trainStart = trainStart
        self.model = model

        self.reset()

        self.action_low = np.array([self.B_min, self.p_min])
        self.action_high = np.array([self.B_max, self.p_max])

        # Define action and observation space
        # They must be gym.spaces objects
        if model == 'dqn':
            self.action_space = spaces.Discrete(2)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents * self.num_states,),
                                            dtype=np.float32)

    def step(self, action):
        obs_n = np.zeros((self.num_agents, self.num_states))  # step后的新观察状态
        reward_n = np.zeros((self.num_agents,))  # step所得reward
        info = {'n': []}  # 信息info
        c1 = 1e0  # 比例因子
        c2 = 1e2  # 比例因子
        self.reward = 0.0
        self.p_sum = 0.0
        self.steps += 1

        # set action for each agent
        for i, agent in enumerate(self.agents):
            if self.episodes < self.trainStart:
                if self.model == 'dqn':
                    if action == 0 and agent.p < self.p_max:
                        agent.p += 1
                    elif agent.p > self.p_min:
                        agent.p -= 1
                else:
                    agent.p = action[i] * (self.p_max - self.p_min) / 2 + (self.p_max + self.p_min) / 2
            else:
                if self.model == 'td3':
                    agent.p = 33.0 * np.random.normal(1, 0.05)
                elif self.model == 'ddpg':
                    agent.p = 40.0 * np.random.normal(1, 0.05)
                else:
                    agent.p = 20.0 * np.random.normal(1, 0.05)
            # agent.B = action[i][1]
            agent.lamda, agent.snr = get_dcp_snc(agent.p, miu=self.mius[i])
            agent.reward = agent.getReward(agent.snr)
            self.reward += agent.reward
            self.p_sum += agent.p
            obs = np.array((agent.Pk, agent.U, agent.B, agent.p, agent.lamda))
            obs_n[i] = obs
            reward_n[i] = agent.reward

        # reward = np.mean(reward_n)
        sf = self.getSystemFair()
        # reward = c1 * reward + c2 * sf  # sum--> mean
        # reward = c1 * reward  # sum--> mean
        reward = c1 * (self.base - self.reward) + c2 * sf
        done = self.getDone()
        tqdm.tqdm.write('episode: ' + str(self.episodes) + ', step: ' +
                        str(self.steps) + ', reward: ' + str(reward) +
                        ', sf: ' + str(sf))
        if done:
            rewards.append(reward)
            Pks.append(np.mean([a.Pk for a in self.agents]))
            Uks.append(np.mean([a.U for a in self.agents]))
            # actions.append(np.mean([a.p for a in self.agents]))
            # targets.append(np.mean([a.reward for a in self.agents]))
        return obs_n.flatten(), reward, done, info

    def getDone(self) -> bool:
        # return bool(self.steps >= self.maxStep or (self.p_sum <= self.p_sumMax and self.reward < self.base))
        return bool(self.steps >= self.maxStep)

    def getSystemFair(self) -> float:
        sf_up = 0.0
        sf_low = 0.0
        for agent in self.agents:
            theta = agent.U / agent.snr
            sf_up += theta
            sf_low += theta ** 2
        sf_up = sf_up ** 2
        sf_low = sf_low * self.num_agents
        return sf_up / sf_low

    def reset(self):
        """
        reset world: set random initial states
        :return:
        """
        obs = np.zeros((self.num_agents, self.num_states))
        self.steps = 0
        self.base = 0.0
        self.episodes += 1

        for i, agent in enumerate(self.agents):
            # agent.Ps.clear()
            # agent.Ps.append(P0)
            # agent.a_s.clear()
            agent.p = 10.0 * np.random.normal(1, 0.05)
            # agent.p = 10.0
            # agent.p = random.uniform(self.p_min, self.p_max)
            # agent.B = random.uniform(self.B_min, self.B_max)
            agent.lamda, agent.snr = get_dcp_snc(agent.p, miu=self.mius[i])
            agent.base = agent.getReward(agent.snr)

            self.base += agent.base
            obs[i] = np.array((agent.Pk, agent.U, agent.B, agent.p, agent.lamda))

        return obs.flatten()  # reward, done, info can't be included
