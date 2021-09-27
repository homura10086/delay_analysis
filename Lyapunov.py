"""
时钟同步精度误差协方差和平均吞吐量的联合优化算法
"""
from math import sqrt, inf, log2, exp, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, integrate
from scipy.stats import bernoulli, norm

# 时延区间估计参数
L = 1e3  # 包长，bits/packet
c = 20  # 信噪比常数
W = 20e6  # 系统带宽 Hz

# 时钟同步参数
xs = []
# ws = []
Ps = []
a_s = []
tao = 0.1     # 采样周期
# p = 1   # 相位噪声
# d = 0   # 固定时延
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
v1 = 1  # 控制比例系数
v2 = 1e-12   # 控制比例系数
bk = 0.01  # Pk虚拟队列的离去过程

x0 = np.array((beta, theta)).reshape(2, 1)  # 时钟状态变量
x_hat = A.dot(x0)
e_avg = x0 - x_hat
P0 = e_avg.dot(e_avg.transpose())

loops = 100  # 卡尔曼滤波器迭代次数


def get_U(lamuda, B):
    fQ = norm.ppf(lamuda)  # 标准正态分布的ICDF
    # print(fQ)
    Bw = (Ts / (Ts + B)) * W  # 考虑保时带的有效带宽
    U = Bw * (log2(1 + c) - sqrt(V / (1e-3 * L)) * fQ)
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


def get_init():
    # 时钟同步模型
    lamuda = 0.01  # p_dcp/数据包到达率

    # print(P0.trace())
    Ps.append(P0)

    Pk, a_avg = get_karman(lamuda)
    B = delt_e + Pk.trace()  # 保时带长度 s
    U = get_U(lamuda, B)
    # print('a0', a_avg)
    base = v1 * Pk.trace() * a_avg - v2 * U
    return Pk, U, lamuda, base


def get_step(lamuda):
    # 执行k步卡尔曼滤波

    # for k in range(loops):
    # mu = np.random.normal(0, sqrt(2 * p), 1)
    # w = np.array((mu, tao * mu)).reshape(2, 1)    # 过程噪声
    # x = A.dot(xs[k]) + w + b
    # xs.append(x)

    # v = np.random.normal(0, sqrt(R), 1)
    # y = C * xs[k] + v
    # z = gamma * y
    # V = 1 - 1 / (1 + C)**2   # 信道色散
    Ps.append(P0)
    # print(P0.trace())
    Pk, a_avg = get_karman(lamuda)
    B = delt_e + Pk.trace()  # 保时带长度 s
    U = get_U(lamuda, B)
    # print('a1', a_avg)
    target = v1 * Pk.trace() * a_avg - v2 * U
    return Pk, U, target, B


if __name__ == '__main__':
    # P1, U0, lamuda, base = get_init()
    lamudas = np.arange(0.01, 1, 0.01)
    Us = []
    targets = []
    Pks = []
    for lamuda in lamudas:
        Pk, U, target, B = get_step(lamuda)
        # print('P0', P1.trace())
        # print('P1', Pk.trace())
        # print('U0', U0)
        # print('U1', U)
        # print('base', base)
        # print('target', target)
        targets.append(target)
        Us.append(U)
        Pks.append(Pk.trace())
    plt.figure(1)
    plt.plot(targets)
    print(np.argmin(targets))
    plt.grid()

    plt.figure(2)
    plt.plot(Us)
    print(np.argmax(Us))
    plt.grid()

    plt.figure(3)
    plt.plot(Pks)
    plt.grid()

    plt.show()
