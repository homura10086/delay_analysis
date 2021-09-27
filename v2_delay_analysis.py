from math import *
from matplotlib import pyplot as plt
from scipy import integrate
from scipy import stats
import numpy as np

#   基于排队论和随机网络演算的时延确定性分析

N0 = pow(10, -160 / 10)  # mW/Hz
NF = pow(10, 1 / 10)  # 接收机噪声系数 1dB
sigma = sqrt(1 / 2)  # 瑞利分布参数
dis = 300  # m
f = 2.6  # Ghz
h_ut = 2  # 天线高度 m
sigma_ls = pow(10, 6 / 10)  # 阴影衰落标准差 6dB
Pl = 13.54 + 39.08 * log10(dis) + 20 * log10(f) - 0.6 * (h_ut - 1.65)   # 路径损耗 UMa NLOS  单位：GHZ, m, dB
G = pow(10, Pl / 10) * sigma_ls  # 比值
# a = 2.5  # 路径损耗因子(2~6)
# Pl = pow(dis, a)     # 比值
# G = Pl * sigma_ls   # 比值
SNR1 = pow(10, 0 / 10)  # =1(比值)
SNR2 = pow(10, 10 / 10)  # =10(比值)
C1 = SNR1
C2 = 1.2
L = 4e3  # bits/packet

t_low = 1e-3  # 时延下界 s
t_up = 6e-3  # 时延上界 s
alpha = 1 / 3   # 时延上界分配因子
t_up_x = 4e-3
# t_up_w = t_up * alpha   # 排队时延的上界（随机网络演算）
t_up_w = t_up - t_up_x
# t_up_x = t_up - t_up_w    # 传输时延的上界

Pt = pow(10, 43 / 10)  # 43dBm/20W
Bs = np.arange(1e5, 3.5e6+1, 1e5)
# B = 1.5e6  # Hz
# Pts = np.arange(1, 3.6e4, 1e3)
lamda = 1e2  # packages/s
# lamdas = np.arange(2e2, 1e3+1, 2e2)
times = 10  # 试验次数

theta_max = 2e-4
# theta = 0   # 随机网络演算参数
# thetas = np.arange(0.01, 1, 0.01)
# thetas = np.array([1e-4, 5e-5, 5e-6, 5e-7, 5e-8])

delay_mcs = []
delay_list = []
p_delay_list = []
p_delay_ncs = []
p_delay_transmits = []

# for Pt in Pts:
for B in Bs:
    lamda_r = stats.poisson.rvs(mu=lamda, size=times, random_state=None)     # 蒙特卡洛随机生成
    z_r = stats.expon.rvs(loc=0, scale=2 * sigma ** 2, size=times, random_state=None)    # 蒙特卡洛随机生成

    C = Pt / (G * N0 * B * NF)  # 比值

    # for theta in thetas:
    #     E_A = exp(lamda * (exp(L * theta) - 1))
    #     exp_s = \
    #         lambda z: exp(- theta * B * log2(1 + C * z)) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    #     E_S = integrate.quad(exp_s, 0, inf)[0]
    #     if E_A * E_S < 1:
    #         theta_max = theta
    #     else:
    #         break
    # print(theta_max)

    # 随机网络演算上边界计算
    # f_upper = \
    #     lambda z: exp(-theta_max * t_up_w * B * log2(1 + C * z)) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    # f_upper_avg = integrate.quad(f_upper, 0, inf)[0]
    # f_upper_nc = \
    #     lambda z: exp(-theta_max * t_up * B * log2(1 + C * z)) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    # f_upper_nc_avg = integrate.quad(f_upper_nc, 0, inf)[0]

    # 香农公式计算服务速率的传输时延一阶矩
    r_std = lambda z: log2(1 + C * z) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    r_std_avg = integrate.quad(r_std, 0, inf)[0]
    x_std_avg = L / (B * r_std_avg)

    #   香农公式计算服务速率的传输时延二阶矩
    x_std_square = \
        lambda z: (L ** 2 * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)) / ((B * log2(1 + C * z)) ** 2)
    x_std_square_avg = integrate.quad(x_std_square, 0, inf)[0]

    #   小信噪比传输时延一阶矩
    r_low = lambda z: C * z * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    r_low_avg = integrate.quad(r_low, 0, SNR1)[0]
    x_low_avg = L / (B * r_low_avg)
    p_low = stats.expon.cdf(SNR1, loc=0, scale=2 * sigma ** 2)  # 小信噪比概率

    #   中信噪比传输时延一阶矩
    r_mid = lambda z: (C1 + C2 * (sqrt(C * z) - sqrt(SNR1))) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    r_mid_avg = integrate.quad(r_mid, SNR1, SNR2)[0]
    x_mid_avg = L / (B * r_mid_avg)
    p_mid = stats.expon.cdf(SNR2, loc=0, scale=2 * sigma ** 2) - p_low  # 中信噪比概率

    #   大信噪比传输时延一阶矩
    r_high = lambda z: log2(C * z) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    r_high_avg = integrate.quad(r_high, SNR2, inf)[0]
    x_high_avg = L / (B * r_high_avg)
    p_high = 1 - stats.expon.cdf(SNR2, loc=0, scale=2 * sigma ** 2)  # 大信噪比概率

    #   总传输时延一阶矩
    # x_avg = x_low_avg * p_low + x_mid_avg * p_mid + x_high_avg * p_high
    x_avg = x_std_avg
    rho = lamda * x_avg  # 业务繁忙率

    #   蒙特卡罗仿真时延计算
    r_mc = B * np.log2(1 + C * z_r)
    x_mc = L / r_mc
    rho_mc = lamda_r * x_mc
    # x_square_mc = np.power((np.mean(x_mc) - x_mc), 2).mean()
    w_mc = (lamda_r * x_std_square_avg) / (2 * (1 - rho_mc))
    w_mc[rho_mc >= 1] = inf
    w_mc[w_mc < 0] = inf

    if rho < 1:
        W = (lamda * x_std_square_avg) / (2 * (1 - rho))  # 平均排队时延
    else:
        W = inf
    if W < 0:
        W = inf

    #   时延抖动区间估计
    # z_up = (pow(2, L / (B * t_up_x)) - 1) / C
    # z_low = (pow(2, L / (B * t_low)) - 1) / C
    # p_upper_x = 1 - stats.expon.cdf(z_up, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
    # p_lower = 1 - stats.expon.cdf(z_low, loc=0, scale=2 * sigma ** 2)  # 小于时延下界概率
    # p_upper_w = 1 - f_upper_avg
    # # p_delay_dcp = (p_upper_x - p_lower) * p_upper_w  # 时延落在上下界区间内的概率
    # p_delay_dcp = (p_upper_x - p_lower)
    # # p_delay = p_upper_w
    # p_delay_nc = 1 - f_upper_nc_avg     # network calculus only

    #   transmit delay only
    # z_up_trans = (pow(2, L / (B * (t_up - W))) - 1) / C
    # z_low_trans = (pow(2, L / (B * (t_low - W))) - 1) / C
    # if t_low - W > 0 and t_up - W > 0:
    #     p_upper_trans = 1 - stats.expon.cdf(z_up_trans, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
    #     p_lower_trans = 1 - stats.expon.cdf(z_low_trans, loc=0, scale=2 * sigma ** 2)  # 小于时延下界概率
    # elif t_low - W < 0 and t_up - W > 0:
    #     p_upper_trans = 1 - stats.expon.cdf(z_up_trans, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
    #     p_lower_trans = 0  # 小于时延下界概率
    # else:
    #     p_upper_trans = p_lower_trans = 0
    # p_delay_trans = p_upper_trans - p_lower_trans  # 时延落在上下界区间内的概率

    # print('二阶矩', '%.3e' % x_std_square_avg)
    # print('传输时延 ', '%.3e' % x_avg)
    # print('排队时延 ', '%.3e' % W)
    # print('概率 ', p_delay)

    delay_list.append((W + x_avg) * 1e3)
    delay_mcs.append((w_mc + x_mc) * 1e3)

    # p_delay_list.append(p_delay_dcp)
    # p_delay_ncs.append(p_delay_nc)
    # p_delay_transmits.append(p_delay_trans)

#   时延和时延概率区间随发射功率的变化
# plt.figure(1)
# plt.xlabel('Transmit Power (W)')
# plt.ylabel('Delay (ms)')
# plt.grid(True)
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.plot(Pts * 1e-3, delay_list, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9,
#          label='average delay')
# plt.scatter(Pts.repeat(times) * 1e-3, delay_mcs, color='red', label='delay', marker='.', alpha=0.9)
# plt.plot(Pts * 1e-3, np.repeat(t_low * 1e3, len(Pts)), label='lower bound')
# plt.plot(Pts * 1e-3, np.repeat(t_up * 1e3, len(Pts)), label='upper bound')
# plt.legend()
# plt.savefig('Fig3(a) DelayVsPower')

# plt.figure(2)
# plt.xlabel('Transmit Power (W)')
# plt.ylabel('Probability')
# plt.grid(True)
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.plot(Pts * 1e-3, p_delay_list, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2,
#          alpha=0.9, label='DCP')
# plt.plot(Pts * 1e-3, p_delay_ncs, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9,
#          label='network calculus only')
# plt.plot(Pts * 1e-3, p_delay_transmits, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2,
#          alpha=0.9, label='transmit delay only')
# plt.legend()
# plt.savefig('Fig3(b) ProbabilityVsPower')


#   时延和时延概率区间随系统带宽的变化

plt.figure(1)
plt.xlabel('Bandwidth (MHz)')
plt.ylabel('Delay (ms)')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.plot(Bs * 1e-6, delay_list, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9,
         label='average delay')
plt.scatter(Bs.repeat(times) * 1e-6, delay_mcs, color='red', label='delay', marker='.', alpha=0.9)
plt.plot(np.hstack((0, Bs)) * 1e-6, np.repeat(t_low * 1e3, len(Bs)+1), label='lower bound')
plt.plot(np.hstack((0, Bs)) * 1e-6, np.repeat(t_up * 1e3, len(Bs)+1), label='upper bound')
plt.legend()
plt.savefig('Fig4(a) DelayVsBw')

# plt.figure(2)
# plt.xlabel('Bandwidth (MHz)')
# plt.ylabel('Probability')
# plt.grid(True)
# Bs = np.hstack((0, Bs))
# p_delay_list = np.hstack((0, p_delay_list))
# p_delay_ncs = np.hstack((0, p_delay_ncs))
# p_delay_transmits = np.hstack((0, p_delay_transmits))
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.plot(Bs * 1e-6, p_delay_list, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2,
#          alpha=0.9, label='DCP')
# plt.plot(Bs * 1e-6, p_delay_ncs, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9,
#          label='network calculus only')
# plt.plot(Bs * 1e-6, p_delay_transmits, linestyle='-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2,
#          alpha=0.9, label='transmit delay only')
# plt.legend()
# plt.savefig('Fig4(b) ProbabilityVsBW')

plt.show()
