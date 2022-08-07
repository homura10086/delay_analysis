"""
时钟同步精度误差协方差和平均吞吐量的联合优化问题建模
PK为标量，吞吐量计算不考虑λ
"""

from math import sqrt, log2
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli, norm
from delay_analysis_QueueAndSNCForLag import L, Bw, get_dcp_snc, t_up, t_low, sigma

# 时延区间估计参数
W = Bw  # 系统带宽 Hz

# 时钟同步参数
Ps = []
a_s = []
k = 0
tao = 0.01   # 采样周期
R = 0.1  # 随机时延协方差
beta = 1.0    # 时钟偏斜
theta = 0.0    # 累计时钟偏移
A = 1 - tao
C = 2.0
q = 5.4e-15  # 过程噪声w的方差

Ts = 1e-3   # 时隙长度 s (scs = 30kHz)
delt_e = 10e-6  # 固定误差 s
v1 = 1e10  # 控制比例系数
v2 = 1e-7   # 控制比例系数
bk = 0.001  # Pk虚拟队列的离去过程

x0 = np.array((beta, theta)).reshape(2, 1)  # 时钟状态变量
x_hat = A * x0
e_avg = x0 - x_hat
P0 = e_avg.transpose().dot(e_avg).item()
loops = 100  # 卡尔曼滤波器迭代次数


def get_U(B: float, snr: float) -> float:
    Bw_e = (Ts / (Ts + B)) * W  # 考虑保时带的有效带宽
    U = Bw_e * log2(1 + snr)
    return U


def get_karman(lamuda: float, loops: int, Ps=Ps, a_s=a_s) -> (float, float):
    Ps.append(P0)
    for k in range(loops):
        # gamma = bernoulli.rvs(p=lamuda, size=1)
        P = A**2 * Ps[k] + q - lamuda * (A * Ps[k] * C)**2 / (C**2 * Ps[k] + R)
        Ps.append(P)
        a = (P - Ps[k]) + bk if Ps[k] >= bk else P  # 虚拟队列到达过程
        a_s.append(a)
    a_avg = np.mean(a_s)
    Pk = Ps.pop()
    Ps.clear()
    # Ps.append(Pk)
    a_s.clear()
    return Pk, a_avg


def get_karmanForSlot(lamuda: float, loops: int, Ps: list, a_s: list) -> (float, float):
    # Ps.append(P0)
    for k in range(loops):
        # gamma = bernoulli.rvs(p=lamuda, size=1)
        P = A**2 * Ps[k] + q - lamuda * (A * Ps[k] * C)**2 / (C**2 * Ps[k] + R)
        Ps.append(P)
        a = (P - Ps[k]) + bk if Ps[k] >= bk else P  # 虚拟队列到达过程
        a_s.append(a)
    a_avg = np.mean(a_s)
    Pk = Ps.pop()
    Ps.clear()
    Ps.append(Pk)
    # a_s.clear()
    return Pk, a_avg


def get_step(Ps: list, a_s: list, lamuda: float, snr: float, loops: int, B=0.0, v1=v1, v2=v2) -> (
        float, float, float, float):
    # 执行loops步卡尔曼滤波
    Pk, a_avg = get_karmanForSlot(lamuda, loops, Ps, a_s)
    if B == 0.0:
        B = delt_e + Pk     # 保时带长度 s
    U = get_U(B, snr)
    target = v1 * Pk * a_avg - v2 * U
    return Pk, U, target, a_avg


def get_stepForParam(lamuda: float, snr: float, loops: int, B=0.0, v1=v1, v2=v2) -> (
        float, float, float, float):
    # 执行loops步卡尔曼滤波
    Pk, a_avg = get_karman(lamuda, loops)
    if B == 0.0:
        B = delt_e + Pk     # 保时带长度 s
    U = get_U(B, snr)
    target = v1 * Pk * a_avg - v2 * U
    return Pk, U, target, a_avg