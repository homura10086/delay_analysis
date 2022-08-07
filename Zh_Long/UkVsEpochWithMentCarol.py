"""
不同 DCP 计算方法下 吞吐量 VS 迭代次数，对比蒙特卡洛随机实验
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from LyapunovComplex import Q, C, A, R, P0, delt_e
from LyapunovSimple import loops as loops_karman, get_U
from delay_analysis_QueueAndSNCForLag import t_low, get_dcp_snc, loops as loops_mc
from delay_analysis_QueueForLag import get_dcp_queue

Pt = 35  # dBm
t_low = t_low   # s


dcp_queue, _ = get_dcp_queue(Pt, t_low)
dcp_snc, snr = get_dcp_snc(Pt, t_low)

# 排队理论
Us_queue = []
P_pre_queue = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre_queue).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre_queue).dot(A.transpose()) + Q - \
        dcp_queue * inv * A.dot(P_pre_queue).dot(C.transpose()).dot(C).dot(P_pre_queue).dot(A.transpose())
    B = delt_e + P.trace()
    U = get_U(B, snr)
    Us_queue.append(U)
    P_pre_queue = P

# 蒙特卡洛
Us_mc = []
for _ in range(loops_mc):
    P_pre_mc = P0
    for _ in range(loops_karman):
        dcp_mc = bernoulli.rvs(p=dcp_snc, size=1)
        inv = np.matrix(C.dot(P_pre_mc).dot(C.transpose()) + R).I.item()
        P = A.dot(P_pre_mc).dot(A.transpose()) + Q - \
            dcp_mc * inv * A.dot(P_pre_mc).dot(C.transpose()).dot(C).dot(P_pre_mc).dot(A.transpose())
        B = delt_e + P.trace()
        U = get_U(B, snr)
        Us_mc.append(U)
        P_pre_mc = P

# 排队理论 + SNC
Us_snc = []
P_pre_snc = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre_snc).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre_snc).dot(A.transpose()) + Q - \
        dcp_snc * inv * A.dot(P_pre_snc).dot(C.transpose()).dot(C).dot(P_pre_snc).dot(A.transpose())
    B = delt_e + P.trace()
    U = get_U(B, snr)
    Us_snc.append(U)
    P_pre_snc = P

# 完全时钟同步模型
Us = []
P_pre = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre).dot(A.transpose()) + Q - \
        inv * A.dot(P_pre).dot(C.transpose()).dot(C).dot(P_pre).dot(A.transpose())
    B = delt_e + P.trace()
    U = get_U(B, snr)
    Us.append(U)
    P_pre = P

plt.rcParams['font.sans-serif'] = ['STZhongsong']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.grid(True)
plt.xlabel('观测步数')
plt.ylabel('U/bps')
marker_size = 3
marker = '.'
plt.plot(np.linspace(0, loops_karman, loops_karman), Us_snc, linestyle='--', marker=marker,
         markersize=marker_size, label='排队论+随机网络演算')
plt.scatter([range(0, loops_karman)] * loops_mc, Us_mc, color='red', marker=marker,
            alpha=0.9, label='蒙特卡洛')
plt.plot(np.linspace(0, loops_karman, loops_karman), Us_queue, linestyle='--', marker=marker,
         markersize=marker_size, label='排队论')
plt.plot(np.linspace(0, loops_karman, loops_karman), Us, linestyle='--', marker=marker,
         markersize=marker_size, label='完全时钟同步')
plt.legend()
plt.savefig('./UVsEpoch')
plt.show()

print(dcp_queue)
print(dcp_snc)
