import random
import numpy as np
from collections import namedtuple

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
from Lyapunov import get_init, get_step
from tqdm import trange
from TDQN import MAX_STEPS, NUM_EPISODES, repeats, col, repeats_ra

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.99
device = 'cpu'
BATCH_SIZE = 32
CAPACITY = 10000
torch.manual_seed(1)
# torch.cuda.manual_seed(1)
random.seed(1)


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


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        # print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)

        else:
            action = torch.tensor([[random.randrange(self.num_actions)]])

        return action


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
            rewards.append(reward)
            Ps.append(P)
            Us.append(U)
            lamudas.append(lamuda)
            Bs.append(B)
            # print('%d Episode | Finished after %d steps | r = %f' % (episode + 1, MAX_STEPS, reward))

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
        pd_data.to_csv('data_DQN.csv', header=is_header, columns=col, index=False, mode=mode)
