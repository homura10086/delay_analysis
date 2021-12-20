"""
基于排队论和随机网络演算的时延确定性分析（无信噪比分段）
"""

from math import sqrt, exp, log2, inf
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import stats
import numpy as np

N0 = pow(10, -160 / 10)  # mW/Hz
sigma = sqrt(1 / 2)  # 瑞利分布参数
dis = 500  # m
# f = 2.6  # Ghz
sigma_ls = pow(10, 6 / 10)  # 阴影衰落标准差 6dB 比值
a = 3  # 路径损耗因子(2~6)
Pl = pow(dis, a)  # 比值
G = 1 / (Pl * sigma_ls)  # 比值
L = 1024 * 8  # 包长  bits/packet

t_low = 0.5e-3  # 时延下界 s
t_up = 1e-3  # 时延上界 s
alpha = 1 / 3  # 时延上界分配因子
# t_up_x = 2e-3   # 传输时延的上界
t_up_w = t_up * alpha   # 排队时延的上界（随机网络演算）
# t_up_w = t_up - t_up_x
t_up_x = t_up - t_up_w

Bw = 1e6  # Hz
lamda = 1e3  # packages/s
h = G / (N0 * Bw)   # 比值
p_delay_list = []


def get_dcp(Pt):
    Pt = pow(10, Pt / 10)  # mW
    snr = Pt * h  # 比值

    # 随机网络演算上边界计算
    theta = 0  # 随机网络演算参数
    E_A = 0
    E_S = 0
    while E_A * E_S <= 1:
        E_A = exp(lamda * (exp(L * theta) - 1) * 1e-9)
        exp_s = \
            lambda z: exp(- theta * Bw * log2(1 + snr * z) * 1e-9) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
        E_S = integrate.quad(exp_s, 0, inf)[0]
        theta += 1e-3
    # print(theta)
    f_upper = \
        lambda z: exp(-theta * t_up_w * Bw * log2(1 + snr * z)) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    f_upper_avg = integrate.quad(f_upper, 0, inf)[0]

    # 传输时延一阶矩
    # r = lambda z: log2(1 + snr * z) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    # r_avg = integrate.quad(r, 0, inf)[0]
    # x_avg = L / (Bw * r_avg)

    # 传输时延二阶矩
    # x_square = \
    #     lambda z: ((L ** 2) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)) / ((Bw * log2(1 + snr * z)) ** 2)
    # x_square_avg = integrate.quad(x_square, 0, inf)[0]
    #
    # rho = lamda * x_avg  # 业务繁忙率

    # if rho < 1:
    #     W = (lamda * x_square_avg) / (2 * (1 - rho))  # 平均排队时延
    # else:
    #     W = inf
    # if W < 0:
    #     W = inf

    #   时延确定性概率估计
    z_up = (pow(2, L / (Bw * t_up_x)) - 1) / snr
    z_low = (pow(2, L / (Bw * t_low)) - 1) / snr
    p_upper_x = 1 - stats.expon.cdf(z_up, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
    p_lower = 1 - stats.expon.cdf(z_low, loc=0, scale=2 * sigma ** 2)  # 小于时延下界概率
    p_upper_w = 1 - f_upper_avg
    p_delay_dcp = (p_upper_x - p_lower) * p_upper_w  # 时延落在上下界区间内的概率
    # p_delay = (p_upper_x - p_lower)

    # print('网络演算上界概率', p_upper_w)
    # print('传输时延区间概率', p_delay)
    # print('DCP概率 ', p_delay_dcp)
    # print('总时延ms', (W + x_avg) * 1e3)
    # print('传输时延ms', x_avg * 1e3)

    return p_delay_dcp, snr
