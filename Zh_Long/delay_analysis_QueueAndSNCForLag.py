"""
基于排队论和随机网络演算的时延确定性分析，拉格朗日算法调用接口
"""

from math import sqrt, exp, log2, inf

from scipy import integrate
from scipy import stats

N0 = pow(10, -160 / 10)  # mW/Hz
sigma = sqrt(1 / 2)  # 瑞利分布参数
dis = 500.0  # m
# f = 2.6  # Ghz
sigma_ls = pow(10, 6 / 10)  # 阴影衰落标准差 6dB, 比值
a = 3.0  # 路径损耗因子(2~6)
Pl = pow(dis, a)  # 比值
G = 1 / (Pl * sigma_ls)  # 比值
L = 1024.0 * 8  # 包长  bits/packet

t_low = 0.5e-3  # 时延下界 s
t_up = 1e-3  # 时延上界 s
alpha = 1 / 3  # 时延上界分配因子
# t_up_x = 2e-3   # 传输时延的上界
t_up_w = t_up * alpha   # 排队时延的上界（随机网络演算）
# t_up_w = t_up - t_up_x
t_up_x = t_up - t_up_w

Bw = 1e6  # Hz
lamda_a = 1e3  # 到达率 packages/s
h = G / (N0 * Bw)   # 比值
times = 100  # 蒙特卡洛单点样本数
loops = 20  # 蒙特卡洛单点实验次数


def get_dcp_snc(Pt: float, t_low=t_low, lamda_a=lamda_a, Bw=Bw) -> (float, float):
    # print("P", Pt)
    Pt = pow(10, Pt / 10)  # mW
    snr = Pt * h  # 比值

    # 随机网络演算上边界计算
    theta = 0.0  # 随机网络演算参数
    E_A = 0.0
    E_S = 0.0
    # for v2/v1
    # index = 1e-1

    # for lamuda
    # index = 1e-2

    # for bandWidth
    index = 1e-4

    # for slot
    # index = 1e-1

    while E_A * E_S <= 1:
        E_A = exp(lamda_a * (exp(L * theta) - 1) * index)
        exp_s = \
            lambda z: exp(- theta * Bw * log2(1 + snr * z) * index) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
        E_S = integrate.quad(exp_s, 0, inf)[0]
        # for lamda
        # theta += 3e-5

        # for bandWidth
        theta += 1e-5

        # for v2/v1
        # theta += 1e-5

        # for slot
        # theta += 1e-4

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

    # 排队时延
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
    p_delay = (p_upper_x - p_lower)
    p_delay_dcp = p_delay * p_upper_w  # 时延落在上下界区间内的概率

    # print('网络演算上界概率', p_upper_w)
    # print('传输时延区间概率', p_delay)
    # print('DCP概率 ', p_delay_dcp)
    # print('总时延ms', (W + x_avg) * 1e3)
    # print('传输时延ms', x_avg * 1e3)
    # print('theta:', theta)
    return p_delay_dcp, snr
