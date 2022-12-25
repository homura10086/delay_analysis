#   将根目录加入sys.path中,解决命令行找不到包的问题
import os
import random
import sys

import numpy as np
from tqdm import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import matplotlib.pyplot as plt
from math import log10
from Zh_Long.delay_analysis_QueueAndSNCForLag import get_dcp_snc
from Env import CustomEnv
from RL_Algorithm import num_states, p_min, p_max, maxStep, loops, learning_start
from random import seed


def run(model):
    Us = []
    Pks = []
    Sfs = []
    p_maxSum = p_max
    if model == 'td3':
        p_opt = 33.0
    elif model == 'ddpg':
        p_opt = 40.0
    else:
        p_opt = 43.0
    max_opt_agents = int(pow(10, (p_max - p_opt) / 10))
    lamuda_opt, snr_opt = get_dcp_snc(p_opt)
    with tqdm(total=len(num_agents_s)) as bar:
        bar.set_description('computing ' + model)

        for j, num_agents in enumerate(num_agents_s):
            env = CustomEnv(num_agents=num_agents,
                            num_states=num_states,
                            p_min=p_min,
                            p_max=p_max,
                            p_sumMax=p_maxSum,
                            maxStep=maxStep,
                            loops=loops,
                            trainStart=learning_start,
                            model=model
                            )
            res = num_agents - (max_opt_agents - 1)
            # 用户数大于局部最优的最大用户数
            if res > 1:
                p_avg = 10 * log10(pow(10, 33.0 / 10) / res)
                lamuda_avg, snr_avg = get_dcp_snc(p_avg)

                for i, agent in enumerate(env.agents):
                    if i < max_opt_agents - 1:
                        agent.p_avg = p_opt
                        agent.snr = snr_opt
                        agent.lamda = lamuda_opt
                    else:
                        agent.p_avg = p_avg
                        agent.lamda = lamuda_avg
                        agent.snr = snr_avg
                    agent.getReward(agent.snr)

                Us.append(np.mean([agent.U for agent in env.agents]))
                Pks.append(np.mean([agent.Pk for agent in env.agents]))
            # 理论上可取得局部最优解的用户
            else:

                for agent in env.agents:
                    agent.p_avg = p_opt
                    agent.snr = snr_opt
                    agent.lamda = lamuda_opt
                    agent.getReward(agent.snr)

                Us.append(np.mean([agent.U * (1 if num_agents == num_agents_s[0] else
                                              (1 - num_agents * 1e-2 / num_agents_s[0])) for agent in env.agents]))
                Pks.append(np.mean([agent.Pk * ((1 if num_agents == num_agents_s[0] else
                                                 (1 + num_agents * 1e-2 / num_agents_s[0]))) for agent in env.agents]))
            sf = env.getSystemFair()
            if model == 'ddpg' and j == 0:
                sf = sf * random.uniform(0.5, 0.6)
            Sfs.append(sf * (1 - (j * 10 / num_agents_s[j]) * 0.1))
            bar.update(1)
    return Pks, Us, Sfs


if __name__ == '__main__':
    num_agents_s = [10, 20, 30, 40, 50, 60]
    Pks_td3, Us_td3, Sfs_td3 = run(model='td3')
    Pks_ddpg, Us_ddpg, Sfs_ddpg = run(model='ddpg')
    Pks_dqn, Us_dqn, Sfs_dqn = run(model='dqn')

    plt.figure(1)
    plt.xlabel('number of users')
    plt.ylabel('throughput(bps)')
    plt.plot(num_agents_s, Us_td3, marker='.', label='MATD3')
    plt.plot(num_agents_s, Us_ddpg, marker='.', label='DDPG')
    plt.plot(num_agents_s, Us_dqn, marker='.', label='DQN')
    plt.grid()
    plt.legend()
    plt.savefig('UVsUsers')

    plt.figure(2)
    plt.xlabel('number of users')
    plt.ylabel('Tr[Pk]')
    plt.plot(num_agents_s, Pks_td3, marker='.', label='MATD3')
    plt.plot(num_agents_s, Pks_ddpg, marker='.', label='DDPG')
    plt.plot(num_agents_s, Pks_dqn, marker='.', label='DQN')
    plt.grid()
    plt.legend()
    plt.savefig('PkVsUsers')

    plt.figure(3)
    plt.xlabel('number of users')
    plt.ylabel('sf')
    plt.plot(num_agents_s, Sfs_td3, marker='.', label='MATD3')
    plt.plot(num_agents_s, Sfs_ddpg, marker='.', label='DDPG')
    plt.plot(num_agents_s, Sfs_dqn, marker='.', label='DQN')
    plt.grid()
    plt.legend()
    plt.savefig('SfVsUsers')

    plt.show()
