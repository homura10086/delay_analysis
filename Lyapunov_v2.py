"""
时钟同步精度误差协方差和平均吞吐量的联合优化问题建模
PK为标量，吞吐量计算公式简化
"""

from math import sqrt, log2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm
from v3_delay_analysis import L, Bw, get_dcp

# 时延区间估计参数
W = Bw  # 系统带宽 Hz

# 时钟同步参数
Ps = []
a_s = []
tao = 0.01   # 采样周期
R = 0.1  # 随机时延协方差
beta = 1    # 时钟偏斜
theta = 0    # 累计时钟偏移
A = 1 - tao
C = 2
q = 5.4e-15  # 过程噪声w的方差

Ts = 1e-3   # 时隙长度 s (scs = 30kHz)
delt_e = 10e-6  # 固定误差 s
v1 = 1e8  # 控制比例系数
v2 = 1e-7   # 控制比例系数
bk = 0.001  # Pk虚拟队列的离去过程

x0 = np.array((beta, theta)).reshape(2, 1)  # 时钟状态变量
x_hat = A * x0
e_avg = x0 - x_hat
P0 = e_avg.transpose().dot(e_avg).item()
loops = 100  # 卡尔曼滤波器迭代次数


def get_U(B, snr):
    # print(fQ)
    Bw_e = (Ts / (Ts + B)) * W  # 考虑保时带的有效带宽
    U = Bw_e * log2(1 + snr)
    # print(U)
    return U


def get_karman(lamuda):
    a_s.clear()
    for k in range(loops):
        # gamma = bernoulli.rvs(p=lamuda, size=1)
        P = A**2 * Ps[k] + q - lamuda * (A * Ps[k] * C)**2 / (C**2 * Ps[k] + R)
        # print(P)
        Ps.append(P)
        a = (P - Ps[k]) + bk if Ps[k] >= bk else P  # 虚拟队列到达过程
        # print(a)
        a_s.append(a)
    a_avg = np.mean(a_s)
    Pk = Ps[len(Ps) - 1]
    print(Ps)
    # print(Pk, a_avg)
    Ps.clear()
    return Pk, a_avg


def get_step(lamuda, snr, B=0):
    # 执行k步卡尔曼滤波
    Ps.append(P0)
    Pk, a_avg = get_karman(lamuda)
    if B == 0:
        B = delt_e + Pk     # 保时带长度 s
    # print(B)
    U = get_U(B, snr)
    target = v1 * Pk * a_avg - v2 * U
    return Pk, U, target


if __name__ == '__main__':
    Pts = np.arange(9, 50, 1)
    Bs = np.arange(5e-5, Ts, 5e-5)
    Us = []
    targets = []
    Pks = []
    # Pt = 25
    for Pt in Pts:
    # for B in Bs:
        lamuda, snr = get_dcp(Pt)
        Pk, U, target = get_step(lamuda, snr)
        # Pk, U, target = get_step(lamuda, snr, B)
        targets.append(target)
        Us.append(U)
        Pks.append(Pk)

    # 发射功率自变量
    plt.figure(1)
    plt.plot(Pts, targets)
    plt.title('target')
    # print(targets)
    print(Pts[np.argmin(targets)])
    plt.grid()

    plt.figure(2)
    plt.plot(Pts, Us)
    plt.title('U')
    # print(Us)
    print(Pts[np.argmax(Us)])
    plt.grid()

    plt.figure(3)
    plt.plot(Pts, Pks)
    plt.title('Pk')
    # print(Pks)
    print(Pts[np.argmin(Pks)])
    plt.grid()

    # 保时带自变量
    # plt.figure(1)
    # plt.plot(Bs, targets)
    # plt.title('target')
    # # print(targets)
    # print(Bs[np.argmin(targets)])
    # plt.grid()
    #
    # plt.figure(2)
    # plt.plot(Bs, Us)
    # plt.title('U')
    # # print(Us)
    # print(Bs[np.argmax(Us)])
    # plt.grid()

    plt.show()
