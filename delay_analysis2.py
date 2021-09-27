from math import *
from matplotlib import pyplot as plt
from scipy import integrate
from sympy import symbols
from scipy import stats

#   基于有效带宽和有效容量理论的时延分析

N0 = pow(10, -174 / 10)  # mW/Hz
NF = pow(10, 1 / 10)  # 接收机噪声系数 1dB
B = 5e6  # Hz
sigma = sqrt(1 / 2)  # 瑞利分布参数
L = 1e3  # bit
dis = 300    # m
f = 2.6     # Ghz
h_ut = 1.5  # 天线高度 m
# h_bs = 25   # 基站高度 m
Pt = pow(10, 46 / 10)  # 46dBm/20W
# 路径损耗 UMa NLOS 阴影衰落标准差6dB 单位：GHZ, m, dB
sigma_ls = pow(10, 6 / 10)    # 阴影衰落标准差 6dB
# Pl = 13.54 + 39.08 * log10(dis) + 20 * log10(f) - 0.6 * (h_ut - 1.65)
a = 2.5   # 路径损耗因子(2~6)
Pl = pow(dis, a) * sigma_ls    # 比值
# G = pow(10, Pl / 10)  # 比值
G = Pl
SNR1 = pow(10, 0 / 10)  # =1(比值)
SNR2 = pow(10, 10 / 10)  # =10(比值)
C1 = SNR1
C2 = 1.2
lamda = 100 * L  # bps
# phi = 0.577  # 欧拉常数γ≈ 0.57721 56649 01532 86060 65120 90082 40243 10421 59335
# N = 10  # 近似样本数
t1 = 5e-3  # 时延下界 ms
t2 = 6e-3  # 时延上界 ms



# plt.plot(delays)
# plt.show()


