from Lyapunov import get_step, get_init
import numpy as np
import random
from TDQN import NUM_EPISODES, repeats, col, repeats_ra
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import trange, tqdm

random.seed(1)

loops = NUM_EPISODES

Ps_list = []
lamudas_list = []
Us_list = []
Bs_list = []


def RA():
    for _ in trange(repeats_ra):
        Ps = []
        lamudas = []
        Us = []
        Bs = []
        # P, U, lamuda, target = get_init()
        for _ in range(loops):
            lamuda = random.uniform(0.01, 0.99)
            P, U, targrt, B = get_step(lamuda)
            Ps.append(P.trace())
            Us.append(U)
            lamudas.append(lamuda)
            Bs.append(B)

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
        pd_data.to_csv('data_RA.csv', header=is_header, columns=col, index=False, mode=mode)


def plot():
    # plot the data
    data = pd.read_csv("data_RA.csv", header=0, usecols=col)
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
    RA()
    # plot()
