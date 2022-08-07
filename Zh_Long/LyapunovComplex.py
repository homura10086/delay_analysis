"""
时钟同步精度误差协方差和平均吞吐量的联合优化问题建模
PK为矩阵，吞吐量计算考虑λ
"""

from math import sqrt, log2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm
from delay_analysis_QueueAndSNCForLag import L, Bw, get_dcp_snc

# 时延区间估计参数
W = Bw  # 系统带宽 Hz

# 时钟同步参数
Ps = []
a_s = []
tao = 0.1     # 采样周期
R = 0.1   # 随机时延协方差
beta = 1    # 时钟偏斜
theta = 0    # 累计时钟偏移
A = np.array(((1, 0), (tao, 1)))
b = np.array((0, -tao)).reshape(2, 1)
C = np.array((0, 2)).reshape(1, 2)
Q = 2.7e-15 * np.identity(2)   # 单位阵, w的协方差矩阵
V = 1   # 信道色散

Ts = 1e-3   # 时隙长度 s (scs = 30kHz)
delt_e = 10e-6  # 固定误差 s
v1 = 1e4  # 控制比例系数
v2 = 1e-6   # 控制比例系数
bk = 0.01  # Pk虚拟队列的离去过程

x0 = np.array((beta, theta)).reshape(2, 1)  # 时钟状态变量
x_hat = A.dot(x0)
e_avg = x0 - x_hat
P0 = e_avg.dot(e_avg.transpose())

loops = 100  # 卡尔曼滤波器迭代次数


def get_U(lamuda, B, c):
    fQ = norm.ppf(lamuda)  # 标准正态分布的ICDF
    # print(fQ)
    Bw_e = (Ts / (Ts + B)) * W  # 考虑保时带的有效带宽
    U = Bw_e * (log2(1 + c) - sqrt(V / L) * fQ)
    # print(U)
    return U


def get_karman(lamuda):
    a_s.clear()
    for k in range(loops):
        inv = np.matrix(C.dot(Ps[k]).dot(C.transpose()) + R).I.item()
        # gamma = bernoulli.rvs(p=lamuda, size=1)
        P = A.dot(Ps[k]).dot(A.transpose()) + Q - \
            lamuda * inv * A.dot(Ps[k]).dot(C.transpose()).dot(C).dot(Ps[k]).dot(A.transpose())
        # print(P.trace())
        Ps.append(P)
        a = (P - Ps[k]).trace() + bk if Ps[k].trace() >= bk else P.trace()  # 虚拟队列到达过程
        # print(a)
        a_s.append(a)
    a_avg = np.mean(a_s)
    Pk = Ps[len(Ps) - 1]
    # print(Pk.trace(), a_avg)
    Ps.clear()
    return Pk, a_avg


def get_step(lamuda, c, B=0):
    # 执行k步卡尔曼滤波
    Ps.append(P0)
    Pk, a_avg = get_karman(lamuda)
    if B == 0:
        B = delt_e + Pk.trace()     # 保时带长度 s
    # print(B)
    U = get_U(lamuda, B, c)
    target = v1 * Pk.trace() * a_avg - v2 * U
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
        lamuda, c = get_dcp_snc(Pt)
        Pk, U, target = get_step(lamuda, c)
        # Pk, U, target = get_step(lamuda, c, B)
        targets.append(target)
        Us.append(U)
        Pks.append(Pk.trace())

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
