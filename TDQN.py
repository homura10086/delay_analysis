import random
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import normal
from torch import nn
from torch import optim
from tqdm import trange
from Lyapunov import get_init, get_step


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
MAX_STEPS = 55
NUM_EPISODES = 100
BATCH_SIZE = 32
CAPACITY = 10000
torch.manual_seed(1)
# torch.cuda.manual_seed(1)
random.seed(1)
device = 'cpu'


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))
        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))
        return output


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.num_states = num_states
        self.memory = ReplayMemory(CAPACITY)
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out).to(device)
        self.target_q_network = Net(n_in, n_mid, n_out).to(device)
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch()
        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval().to('cpu')
            with torch.no_grad():
                action = self.main_q_network(state.to('cpu')).max(1)[1].view(1, 1)
        else:
            action = torch.tensor([[random.randrange(self.num_actions)]])
        return action.to(device)

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.int64)
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        return expected_state_action_values

    def update_main_q_network(self):
        self.main_q_network.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


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
        # print(self.P.trace())
        # print(self.B)
        state_next = np.array((self.P.trace(), self.U, self.lamuda))  # (3,)
        return state_next, reward, B

    def reset(self):
        self.P, self.U, self.lamuda, self.base = get_init()
        self.state_space = np.array((self.P.trace(), self.U, self.lamuda))  # (3,)
        self.action_space = np.array(self.lamuda)  # (1,)
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
            state = torch.from_numpy(state).type(torch.float32).to(device)
            state = torch.unsqueeze(state, 0).to(device)
            for step in range(MAX_STEPS):
                action = self.agent.get_action(state, episode)
                state_next, reward, B = self.env.step(action.to('cpu'))
                P = state_next[0]
                U = state_next[1]
                lamuda = state_next[2]
                state_next = torch.from_numpy(state_next).type(torch.float32).to(device)
                state_next = torch.unsqueeze(state_next, 0).to(device)
                self.agent.memorize(state, action.to(device), state_next,
                                    torch.tensor([reward], dtype=torch.float32).to(device))
                self.agent.update_q_function()
                state = state_next.to(device)
            if episode % 2 == 0:
                self.agent.update_target_q_function()
            rewards.append(reward)
            Ps.append(P)
            Us.append(U)
            lamudas.append(lamuda)
            Bs.append(B)
            print('%d Episode | Finished after %d steps | r = %f' % (episode + 1, MAX_STEPS, reward))

        # 画出训练过程
        plt.figure(1)
        plt.grid()
        plt.ylabel('reward')
        plt.xlabel('epoch')
        plt.plot(rewards)

        plt.figure(2)
        plt.grid()
        plt.ylabel('P')
        plt.xlabel('epoch')
        plt.plot(Ps)

        plt.figure(3)
        plt.grid()
        plt.ylabel('U')
        plt.xlabel('epoch')
        plt.plot(Us)

        plt.figure(4)
        plt.grid()
        plt.ylabel('lamuda')
        plt.xlabel('epoch')
        plt.plot(lamudas)

        plt.figure(5)
        plt.grid()
        plt.ylabel('B')
        plt.xlabel('epoch')
        plt.plot(Bs)

        plt.show()
        return Ps, Us, lamudas, Bs


repeats = 1
repeats_ra = 10
Ps_list = []
lamudas_list = []
Us_list = []
Bs_list = []
col = ('Ps', 'Us', 'lamudas', 'Bs')


def plot():
    # plot the data

    data_tdqn = pd.read_csv("data_TDQN.csv", header=0, usecols=col)
    Ps_tdqn = np.array(data_tdqn['Ps']).reshape(NUM_EPISODES, repeats).mean(1)
    Us_tdqn = np.array(data_tdqn['Us']).reshape(NUM_EPISODES, repeats).mean(1)
    lamudas_tdqn = np.array(data_tdqn['lamudas']).reshape(NUM_EPISODES, repeats).mean(1)
    Bs_tdqn = np.array(data_tdqn['Bs']).reshape(NUM_EPISODES, repeats).mean(1)

    data_dqn = pd.read_csv("data_DQN.csv", header=0, usecols=col)
    Ps_dqn = np.array(data_dqn['Ps']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Us_dqn = np.array(data_dqn['Us']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    lamudas_dqn = np.array(data_dqn['lamudas']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Bs_dqn = np.array(data_dqn['Bs']).reshape(NUM_EPISODES, repeats_ra).mean(1)

    data_q = pd.read_csv("data_Q.csv", header=0, usecols=col)
    Ps_q = np.array(data_q['Ps']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Us_q = np.array(data_q['Us']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    lamudas_q = np.array(data_q['lamudas']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Bs_q = np.array(data_q['Bs']).reshape(NUM_EPISODES, repeats_ra).mean(1)

    data_ra = pd.read_csv("data_RA.csv", header=0, usecols=col)
    Ps_ra = np.array(data_ra['Ps']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Us_ra = np.array(data_ra['Us']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    lamudas_ra = np.array(data_ra['lamudas']).reshape(NUM_EPISODES, repeats_ra).mean(1)
    Bs_ra = np.array(data_ra['Bs']).reshape(NUM_EPISODES, repeats_ra).mean(1)

    plt.rcParams['font.sans-serif'] = ['STZhongsong']
    plt.rcParams['axes.unicode_minus'] = False
    marker_size = 5

    plt.figure(1)
    plt.grid(True)
    plt.ylabel('Tr[P]', fontsize=12)
    plt.xlabel('迭代次数', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.plot(Ps_tdqn, marker='.', markersize=marker_size, alpha=0.9, label='3DQN')
    plt.plot(Ps_dqn, marker='.', markersize=marker_size, alpha=0.9, label='DQN')
    plt.plot(Ps_ra, marker='.', markersize=marker_size, alpha=0.9, label='RA')
    plt.plot(Ps_q, marker='.', markersize=marker_size, alpha=0.9, label='Q-learning')
    plt.legend(loc='best', fancybox=True, shadow=False, frameon=True)
    plt.savefig('fig6')

    plt.figure(2)
    plt.grid(True)
    plt.ylabel('U(bps)', fontsize=12)
    plt.xlabel('迭代次数', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.plot(Us_tdqn, marker='.', markersize=marker_size, alpha=0.9, label='3DQN')
    plt.plot(Us_dqn, marker='.', markersize=marker_size, alpha=0.9, label='DQN')
    plt.plot(Us_ra, marker='.', markersize=marker_size, alpha=0.9, label='RA')
    plt.plot(Us_q, marker='.', markersize=marker_size, alpha=0.9, label='Q-learning')
    plt.legend(loc='best', fancybox=True, shadow=False, frameon=True)
    plt.savefig('fig7')

    # plt.figure(3)
    # plt.grid(True)
    # plt.ylabel('λ', fontsize=12)
    # plt.xlabel('迭代次数', fontsize=12)
    # plt.tick_params(axis='both', which='major', labelsize=10)
    # plt.plot(lamudas_tdqn, markersize=marker_size, marker='.', alpha=0.9, label='3DQN')
    # plt.plot(lamudas_dqn, markersize=marker_size, marker='.', alpha=0.9, label='DQN')
    # plt.plot(lamudas_ra, markersize=marker_size, marker='.', alpha=0.9, label='RA')
    # plt.plot(lamudas_q, markersize=marker_size, marker='.', alpha=0.9, label='Q-learning')
    # plt.legend(loc='best', fancybox=True, shadow=False, frameon=True)
    # plt.savefig('fig6')
    #
    # plt.figure(4)
    # plt.grid(True)
    # plt.ylabel('B(s)', fontsize=12)
    # plt.xlabel('迭代次数', fontsize=12)
    # plt.tick_params(axis='both', which='major', labelsize=10)
    # plt.plot(Bs_tdqn, markersize=marker_size, marker='.', alpha=0.9, label='3DQN')
    # plt.plot(Bs_dqn, markersize=marker_size, marker='.', alpha=0.9, label='DQN')
    # plt.plot(Bs_ra, markersize=marker_size, marker='.', alpha=0.9, label='RA')
    # plt.plot(Bs_q, markersize=marker_size, marker='.', alpha=0.9, label='Q-learning')
    # plt.legend(loc='best', fancybox=True, shadow=False, frameon=True)
    # plt.savefig('fig7')

    plt.show()


def train_and_save():
    for _ in range(repeats):
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
        pd_data.to_csv('data_TDQN.csv', header=is_header, columns=col, index=False, mode=mode)


if __name__ == '__main__':
    # train_and_save()
    plot()
