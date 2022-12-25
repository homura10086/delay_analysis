"""
不同时钟同步精度模型，吞吐量/时钟同步精度 VS 数据包到达率
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from Lag_kktAndLinear import getSolution, getSolutionForParma


# from UkAndPkVsSlot import col


def saveData(modelName='snc', algorithmName='lag'):
    # save the data
    data_name = 'data_UkAndPkVsArrival_' + modelName + '_' + algorithmName + '.csv'
    for i, (Pk, Uk) in enumerate(zip(Ps_lag, Us_lag)):
        data_save = np.array((Pk, Uk)).reshape(1, 2)
        pd_data = pd.DataFrame(data_save, columns=col)
        mode = 'w' if i == 0 else 'a'
        is_header = True if i == 0 else False
        pd_data.to_csv(data_name, header=is_header, columns=col, index=False, mode=mode)


def plotData():
    plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题

    data_snc_lag = pd.read_csv('data_UkAndPkVsArrival_snc_lag.csv', header=0, usecols=col)
    Pks_snc_lag = np.array(data_snc_lag['Pk'])
    Uks_snc_lag = np.array(data_snc_lag['Uk'])

    data_snc_ga = pd.read_csv('data_UkAndPkVsArrival_snc_ga.csv', header=0, usecols=col)
    Pks_snc_ga = np.array(data_snc_ga['Pk'])
    Uks_snc_ga = np.array(data_snc_ga['Uk'])

    data_queue_lag = pd.read_csv('data_UkAndPkVsArrival_queue_lag.csv', header=0, usecols=col)
    Pks_queue_lag = np.array(data_queue_lag['Pk'])
    Uks_queue_lag = np.array(data_queue_lag['Uk'])

    data_queue_ga = pd.read_csv('data_UkAndPkVsArrival_queue_ga.csv', header=0, usecols=col)
    Pks_queue_ga = np.array(data_queue_ga['Pk'])
    Uks_queue_ga = np.array(data_queue_ga['Uk'])

    fig = plt.figure(1)
    ax1 = fig.add_subplot()
    lns1 = ax1.plot(lambdas, Pks_snc_lag, label='确定性不完全模型（拉格朗日-KKT求解）', marker='.', linestyle='--')
    lns2 = ax1.plot(lambdas, Pks_queue_lag, label='不完全模型（拉格朗日-KKT求解）', marker='*', linestyle='--')
    lns3 = ax1.plot(lambdas, Pks_snc_ga, label='确定性不完全模型（遗传算法求解）', marker='^', linestyle='--')
    lns4 = ax1.plot(lambdas, Pks_queue_ga, label='不完全模型（遗传算法求解）', marker='>', linestyle='--')
    ax1.set_ylabel('P')
    ax1.set_xlabel('包到达率（个/秒）')
    ax1.set_xscale('log')  # 对数坐标轴
    ax1.invert_yaxis()  # y轴反向
    ax1.grid()

    lns_p = lns1 + lns2 + lns3 + lns4
    labs = [ln.get_label() for ln in lns_p]
    plt.legend(lns_p, labs)
    plt.savefig('PkVsLambda')
    plt.show()

    fig = plt.figure(2)
    ax2 = fig.add_subplot()
    lns5 = ax2.plot(lambdas, Uks_snc_lag, label='确定性不完全模型（拉格朗日-KKT求解）', marker='.', linestyle='--')
    lns6 = ax2.plot(lambdas, Uks_queue_lag, label='不完全模型（拉格朗日-KKT求解）', marker='*', linestyle='--')
    lns7 = ax2.plot(lambdas, Uks_snc_ga, label='确定性不完全模型（遗传算法求解）', marker='^', linestyle='--')
    lns8 = ax2.plot(lambdas, Uks_queue_ga, label='不完全模型（遗传算法求解）', marker='>', linestyle='--')
    ax2.set_ylabel('U/bps')
    ax2.set_xlabel('包到达率（个/秒）')
    ax2.set_xscale('log')  # 对数坐标轴
    ax2.grid()

    lns_u = lns5 + lns6 + lns7 + lns8
    labs = [ln.get_label() for ln in lns_u]
    plt.legend(lns_u, labs)
    plt.savefig('UkVsLambda')
    plt.show()


def getUkAndPk(loops: int, modelName: str, algorithmName: str):
    with tqdm(total=len(lambdas)) as bar:
        bar.set_description('Computing')
        for lamda in lambdas:
            # 拉格朗日算法求解结果
            Uk_lag, Pk_lag = getSolutionForParma(
                loops=loops, lamda_a=lamda, modelName=modelName, algorithmName=algorithmName)
            Ps_lag.append(Pk_lag)
            Us_lag.append(Uk_lag)

            bar.update(1)


if __name__ == '__main__':
    Us_lag = []
    Ps_lag = []
    col = ('Pk', 'Uk')
    lambdas = np.logspace(0, 4, 20)
    loops = 100  # 优化算法迭代的时隙间隔

    'queue: 排队论模型'
    'snc：随机网络演算模型'
    # modelName = 'snc'
    modelName = 'queue'

    'lag：拉格朗日-KKT启发式算法'
    'ga：遗传算法'
    # algorithmName = 'lag'
    algorithmName = 'ga'

    # getUkAndPk(loops, modelName, algorithmName)
    # saveData(modelName, algorithmName)
    plotData()
