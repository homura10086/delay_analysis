#   将根目录加入sys.path中,解决命令行找不到包的问题
import os
import sys

import pandas as pd

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
from Env import CustomEnv, rewards, Pks, Uks, actions
from stable_baselines3 import TD3, DDPG, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt


def learn(modelName: str):
    env = CustomEnv(num_agents=num_agents,
                    num_states=num_states,
                    p_min=p_min,
                    p_max=p_max,
                    p_sumMax=p_maxSum,
                    maxStep=maxStep,
                    loops=loops,
                    trainStart=trainStart,
                    model=modelName,
                    )
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    # The noise objects for TD3
    n_actions = env.action_space.shape
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # envM = Monitor(env=env,
    #                filename='./model',
    #                )
    if modelName == 'td3':
        model = TD3(policy="MlpPolicy",
                    env=env,
                    action_noise=action_noise,
                    verbose=verbose,
                    device="cpu",
                    learning_rate=learning_rate,
                    learning_starts=learning_start,
                    batch_size=batch_size,
                    seed=1,
                    )
    elif modelName == 'ddpg':
        model = DDPG(policy="MlpPolicy",
                     env=env,
                     action_noise=action_noise,
                     verbose=verbose,
                     device="cpu",
                     learning_rate=learning_rate,
                     learning_starts=learning_start,
                     batch_size=batch_size,
                     seed=1,
                     )
    else:
        model = DQN(policy="MlpPolicy",
                    env=env,
                    verbose=verbose,
                    device="cpu",
                    learning_starts=learning_start,
                    seed=1,
                    )
    model.learn(total_timesteps=total_timesteps,
                log_interval=log_interval,
                )
    model.save(modelName)


def saveData(modelName: str):
    # save the data
    for i, (reward, Pk, Uk) in enumerate(zip(rewards, Pks, Uks)):
        data_save = np.array((reward, Pk, Uk)).reshape(1, 3)
        pd_data = pd.DataFrame(data_save, columns=col)
        mode = 'w' if i == 0 else 'a'
        is_header = True if i == 0 else False
        pd_data.to_csv(modelName + '.csv', header=is_header, columns=col, index=False, mode=mode)


def plotAndSaveFig():
    # for Zh plot
    plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题

    data_td3 = pd.read_csv('td3.csv', header=0, usecols=col)[:201]
    rewards_td3 = np.array(data_td3['reward'])
    Pks_td3 = np.array(data_td3['Pk'])
    Uks_td3 = np.array(data_td3['Uk'])

    data_ddpg = pd.read_csv('ddpg.csv', header=0, usecols=col)[:201]
    rewards_ddpg = np.array(data_ddpg['reward'])
    Pks_ddpg = np.array(data_ddpg['Pk'])
    Uks_ddpg = np.array(data_ddpg['Uk'])

    data_dqn = pd.read_csv('dqn.csv', header=0, usecols=col)[:201]
    rewards_dqn = np.array(data_dqn['reward'])
    Pks_dqn = np.array(data_dqn['Pk'])
    Uks_dqn = np.array(data_dqn['Uk'])

    # plot the data
    plt.figure(1)
    plt.plot(rewards_td3, label='MATD3', linestyle='-')
    plt.plot(rewards_ddpg, label='DDPG', linestyle='-')
    plt.plot(rewards_dqn, label='DQN', linestyle='-')
    plt.title("reward")
    plt.grid()
    plt.legend()
    plt.savefig('reward')
    # plt.figure(2)
    # plt.plot(actions)
    # plt.title("p")
    # plt.grid()

    # plt.figure(3)
    # plt.plot(targets)
    # plt.title("target")
    # plt.grid()

    plt.figure(4)
    plt.plot(Pks_td3, label='MATD3', linestyle='-')
    plt.plot(Pks_ddpg, label='DDPG', linestyle='-')
    plt.plot(Pks_dqn, label='DQN', linestyle='-')
    plt.title("Tr(E[Pk])")
    plt.grid()
    plt.legend()
    plt.savefig('PK')

    plt.figure(5)
    plt.plot(Uks_td3, label='MATD3', linestyle='-')
    plt.plot(Uks_ddpg, label='DDPG', linestyle='-')
    plt.plot(Uks_dqn, label='DQN', linestyle='-')
    plt.title("throughput")
    plt.grid()
    plt.legend()
    plt.savefig('U')

    plt.show()


num_agents = 4
num_states = 5
p_min = 10.0
p_max = 50.0
p_maxSum = p_max * num_agents
maxStep = 20
loops = 100
maxEpisodes = 100
trainStart = 100
learning_start = maxStep * 100
learning_rate = 1e-3
log_interval = 1
batch_size = learning_start
total_timesteps = maxStep * maxEpisodes + learning_start
verbose = 0
col = ('reward', 'Pk', 'Uk')
if __name__ == '__main__':
    # modelName = 'td3'
    # modelName = 'ddpg'
    # modelName = 'dqn'
    # learn(modelName=modelName)
    # saveData(modelName=modelName)
    plotAndSaveFig()
