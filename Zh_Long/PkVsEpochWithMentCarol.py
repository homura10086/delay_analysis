"""
不同 DCP 计算方法下 时钟同步模型精度 VS 迭代次数，对比蒙特卡洛随机实验
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from LyapunovComplex import Q, C, A, R, P0
from LyapunovSimple import loops as loops_karman
from delay_analysis_QueueAndSNCForLag import t_low, get_dcp_snc, loops as loops_mc
from delay_analysis_QueueForLag import get_dcp_queue

Pt = 35  # dBm
t_low = t_low   # s


dcp_queue, _ = get_dcp_queue(Pt, t_low)
dcp_snc, _ = get_dcp_snc(Pt, t_low)

# 排队理论
Ps_queue = []
P_pre_queue = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre_queue).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre_queue).dot(A.transpose()) + Q - \
        dcp_queue * inv * A.dot(P_pre_queue).dot(C.transpose()).dot(C).dot(P_pre_queue).dot(A.transpose())
    Ps_queue.append(P.trace())
    P_pre_queue = P

# 蒙特卡洛
Ps_mc = []
for _ in range(loops_mc):
    P_pre_mc = P0
    for _ in range(loops_karman):
        dcp_mc = bernoulli.rvs(p=dcp_snc, size=1)
        inv = np.matrix(C.dot(P_pre_mc).dot(C.transpose()) + R).I.item()
        P = A.dot(P_pre_mc).dot(A.transpose()) + Q - \
            dcp_mc * inv * A.dot(P_pre_mc).dot(C.transpose()).dot(C).dot(P_pre_mc).dot(A.transpose())
        Ps_mc.append(P.trace())
        P_pre_mc = P

# 排队理论 + SNC
Ps_snc = []
P_pre_snc = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre_snc).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre_snc).dot(A.transpose()) + Q - \
        dcp_snc * inv * A.dot(P_pre_snc).dot(C.transpose()).dot(C).dot(P_pre_snc).dot(A.transpose())
    Ps_snc.append(P.trace())
    P_pre_snc = P

# 完全时钟同步模型
Ps = []
P_pre = P0
for _ in range(loops_karman):
    inv = np.matrix(C.dot(P_pre).dot(C.transpose()) + R).I.item()
    P = A.dot(P_pre).dot(A.transpose()) + Q - \
        inv * A.dot(P_pre).dot(C.transpose()).dot(C).dot(P_pre).dot(A.transpose())
    Ps.append(P.trace())
    P_pre = P

plt.rcParams['font.sans-serif'] = ['STZhongsong']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.grid(True)
plt.xlabel('观测步数')
plt.ylabel('P')
marker_size = 5
plt.plot(np.linspace(0, loops_karman, loops_karman), Ps_snc, linestyle='--', marker='+',
         markersize=marker_size, label='确定性不完全时钟同步模型')
plt.scatter([range(0, loops_karman)] * loops_mc, Ps_mc, color='red', marker='.',
            alpha=0.9, label='蒙特卡洛随机实验')
plt.plot(np.linspace(0, loops_karman, loops_karman), Ps_queue, linestyle='--', marker='.',
         markersize=marker_size, label='不完全时钟同步模型')
plt.plot(np.linspace(0, loops_karman, loops_karman), Ps, linestyle='--', marker='^',
         markersize=marker_size, label='完全时钟同步模型')
plt.legend()
plt.savefig('./PkVsEpoch')
plt.show()

# print(dcp_queue)
# print(dcp_snc)
