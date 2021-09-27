from math import *
from sympy import *
# from sympy.abc import z
from matplotlib import pyplot as plt

# z = symbols('z')
# snr1 = symbols('snr1')
# snr2 = symbols('snr2')

snr1 = 1    # 小信噪比门限
# snr2 = 15   # 大信噪比门限
c1 = snr1   # 拟合参数
snr2 = 10    # 大信噪比门限
# c2 = (log2(snr2) - c1) / (sqrt(snr2) - sqrt(snr1))  # 拟合参数
c2 = 1.2
R_mids = []
R_lows = []
R_highs = []
Rs = []
zs = []

z = 0.1
for _ in range(1000):
    R_low = z
    R_mid = c1 + c2 * (sqrt(z) - sqrt(snr1))
    R_high = log2(z)
    R = log2(1 + z)
    Rs.append(R)
    R_lows.append(R_low)
    R_highs.append(R_high)
    R_mids.append(R_mid)
    zs.append(z)
    z += 0.1

plt.plot(zs, Rs, linestyle='-', markerfacecolor='none', markeredgewidth=1, lw=2,
         alpha=0.9, label='Shannon capacity')
# plt.plot(zs, R_lows, label='r')
plt.plot(zs, R_mids, linestyle='-', markerfacecolor='none', markeredgewidth=1, lw=2,
         alpha=0.9, label='Fitting curve')
# plt.plot(zs, R_highs, label='log2(r)')
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlim(1, 10)
plt.ylim(0, 5)
plt.xlabel('SNR (ratio)')
plt.ylabel('Channel Capacity (bps/Hz)')
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fancybox=True, shadow=False, frameon=True, fontsize=12)
plt.savefig('Fig2 MidSNRChannelFitting.png')
plt.show()
