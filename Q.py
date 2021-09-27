import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import trange

from TDQN import NUM_EPISODES, MAX_STEPS, repeats, col, repeats_ra
from Lyapunov import get_init, get_step
import pandas as pd

random.seed(1)
NUM_DIZITIZED = 100   # 离散化数
GAMMA = 0.99
ETA = 0.5


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next)

    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action


class Brain:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.q_table = np.random.uniform(low=0, high=1, size=(
            NUM_DIZITIZED**num_states, num_actions))

    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def digitize_state(self, state):
        P = state[0]
        U = state[1]
        lamuda = state[2]
        digitized = [
            np.digitize(P, bins=self.bins(0, 0.01, NUM_DIZITIZED)),
            np.digitize(U, bins=self.bins(0, 6e7, NUM_DIZITIZED)),
            np.digitize(lamuda, bins=self.bins(0, 1, NUM_DIZITIZED)),
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        state = self.digitize_state(observation)
        state_next = self.digitize_state(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        self.q_table[state, action] = self.q_table[state, action] + \
            ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])

    def decide_action(self, state, episode):
        state = self.digitize_state(state)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[state][:])
        else:
            action = np.random.choice(self.num_actions)
        return action


class Env:
    def __init__(self):
        self.reset()

    def step(self, action):
        if action == 0 and self.lamuda < 0.99:
            self.lamuda += 0.01
        elif action == 1 and self.lamuda > 0.01:
            self.lamuda -= 0.01
        self.P, self.U, self.targrt, B = get_step(self.lamuda)
        reward = (self.base - self.targrt) * 1e6
        state_next = np.array((self.P.trace(), self.U, self.lamuda))    # (3,)
        return state_next, reward, B

    def reset(self):
        self.P, self.U, self.lamuda, self.base = get_init()
        self.state_space = np.array((self.P.trace(), self.U, self.lamuda))  # (3,)
        self.action_space = np.array(self.lamuda)     # (1,)
        return self.state_space


class Environment:

    def __init__(self):
        self.env = Env()
        num_states = self.env.state_space.shape[0]
        num_actions = self.env.action_space.size * 2
        self.agent = Agent(num_states, num_actions)

    def run(self):
        reward = 0.0
        P = 0.0
        U = 0.0
        lamuda = 0.0
        B = 0.0
        rewards = []
        Ps = []
        lamudas = []
        Us = []
        Bs = []
        for episode in trange(NUM_EPISODES):
            state = self.env.reset()

            for step in range(MAX_STEPS):

                action = self.agent.get_action(state, episode)

                state_next, reward, B = self.env.step(action)
                P = state_next[0]
                U = state_next[1]
                lamuda = state_next[2]
                self.agent.update_Q_function(state, action, reward, state_next)

                state = state_next

                # if step == MAX_STEPS - 1:
                #     print('%d Episode | Finished after %d steps | r = %f' % (episode + 1, step + 1, reward))
            rewards.append(reward)
            Ps.append(P)
            Us.append(U)
            lamudas.append(lamuda)
            Bs.append(B)
        # 画出训练过程
        # plt.figure(1)
        # plt.grid()
        # plt.ylabel('reward')
        # plt.xlabel('epoch')
        # plt.plot(rewards)
        #
        # plt.figure(2)
        # plt.grid()
        # plt.ylabel('P')
        # plt.xlabel('epoch')
        # plt.plot(Ps)
        #
        # plt.figure(3)
        # plt.grid()
        # plt.ylabel('U')
        # plt.xlabel('epoch')
        # plt.plot(Us)
        #
        # plt.figure(4)
        # plt.grid()
        # plt.ylabel('lamuda')
        # plt.xlabel('epoch')
        # plt.plot(lamudas)
        #
        # plt.figure(5)
        # plt.grid()
        # plt.ylabel('B')
        # plt.xlabel('epoch')
        # plt.plot(Bs)
        #
        # plt.show()
        return Ps, Us, lamudas, Bs


Ps_list = []
lamudas_list = []
Us_list = []
Bs_list = []


def plot():
    # plot the data
    data = pd.read_csv("data_Q.csv", header=0, usecols=col)
    Ps = np.array(data['Ps']).reshape(NUM_EPISODES, repeats).mean(1)
    Us = np.array(data['Us']).reshape(NUM_EPISODES, repeats).mean(1)
    lamudas = np.array(data['lamudas']).reshape(NUM_EPISODES, repeats).mean(1)
    Bs = np.array(data['Bs']).reshape(NUM_EPISODES, repeats).mean(1)

    plt.figure(1)
    plt.grid()
    plt.ylabel('P')
    plt.xlabel('epoch')
    plt.plot(Ps)

    plt.figure(2)
    plt.grid()
    plt.ylabel('U')
    plt.xlabel('epoch')
    plt.plot(Us)

    plt.figure(3)
    plt.grid()
    plt.ylabel('lamuda')
    plt.xlabel('epoch')
    plt.plot(lamudas)

    plt.figure(4)
    plt.grid()
    plt.ylabel('B')
    plt.xlabel('epoch')
    plt.plot(Bs)

    plt.show()


if __name__ == '__main__':
    for _ in range(repeats_ra):
        Ps, Us, lamudas, Bs = Environment().run()
        Ps_list.append(Ps)
        Us_list.append(Us)
        lamudas_list.append(lamudas)
        Bs_list.append(Bs)

    # save the data
    for i, (Ps, Us, lamudas, Bs) in enumerate(zip(Ps_list, Us_list, lamudas_list, Bs_list)):
        data_save = np.array((Ps, Us, lamudas, Bs)).transpose()
        pd_data = pd.DataFrame(data_save, columns=col)
        mode = 'w' if i == 0 else 'a'
        is_header = True if i == 0 else False
        pd_data.to_csv('data_Q.csv', header=is_header, columns=col, index=False, mode=mode)

    # plot()
