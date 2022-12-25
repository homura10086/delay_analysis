"""
不同时钟同步精度模型，系统吞吐量/时钟同步精度 VS 求解算法迭代步数/时隙数？
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Lag_kktAndLinear import getSolution, Pks, Uks


def saveData(modelName='snc'):
    # save the data
    data_name = 'data_UkAndPkVsSlot_' + modelName + '.csv'
    for i, (Pk, Uk) in enumerate(zip(Pks, Uks)):
        data_save = np.array((Pk, Uk)).reshape(1, 2)
        pd_data = pd.DataFrame(data_save, columns=col)
        mode = 'w' if i == 0 else 'a'
        is_header = True if i == 0 else False
        pd_data.to_csv(data_name, header=is_header, columns=col, index=False, mode=mode)


def plotData():
    plt.rcParams['font.sans-serif'] = ['STZhongsong']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 该语句解决图像中的“-”负号的乱码问题

    data_snc = pd.read_csv('data_UkAndPkVsSlot_snc.csv', header=0, usecols=col)
    Pks_snc = np.array(data_snc['Pk'])
    Uks_snc = np.array(data_snc['Uk'])

    fig = plt.figure(1)
    ax1 = fig.add_subplot()
    lns1 = ax1.plot(Pks_snc, label='时钟同步精度', marker='^', linestyle='--', color='r')
    ax1.set_ylabel('P')
    ax1.set_xlabel('迭代次数')
    ax1.grid()

    ax2 = ax1.twinx()
    lns2 = ax2.plot(Uks_snc, label='吞吐量', marker='.', linestyle='--', color='b')
    ax2.set_ylabel('U/bps')
    ax2.grid()

    lns = lns1 + lns2
    labs = [ln.get_label() for ln in lns]
    plt.legend(lns, labs)
    plt.savefig('PkAndUkVsSlot')
    plt.show()


if __name__ == '__main__':
    '执行优化算法并保存结果'

    'snc：随机网络演算模型'
    modelName = 'snc'

    loops = 10  # 优化算法迭代的时隙间隔
    N = 30  # 算法迭代次数
    col = ('Pk', 'Uk')

    getSolution(N=N,
                loops=loops,
                modelName=modelName,
                )

    saveData(modelName)

    plotData()
