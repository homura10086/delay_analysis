"""
不同时钟同步模型吞吐量随λ的变化曲线
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from Lyapunov import get_U, delt_e

# 时钟同步参数
tao = 0.1  # 采样周期
R = 0.1  # 随机时延协方差
beta = 1  # 时钟偏斜
theta = 0  # 累计时钟偏移
A = np.array(((1, 0), (tao, 1)))
b = np.array((0, -tao)).reshape(2, 1)
C = np.array((0, 2)).reshape(1, 2)
Q = 2.7e-15 * np.identity(2)  # 单位阵, w的协方差矩阵

x0 = np.array((beta, theta)).reshape(2, 1)  # 时钟状态变量
x_hat = A.dot(x0)
e_avg = x0 - x_hat
P0 = e_avg.dot(e_avg.transpose())

loops_mc = 10  # 蒙特卡洛实验次数
loops_kar = 100  # 卡尔曼滤波器迭代次数

lamudas = np.arange(0.01, 1, 0.01)

P_mc = np.zeros((2, 2))
P_avg = np.zeros((2, 2))
P_cmp = np.zeros((2, 2))
# P_bwq = np.zeros((2, 2))

Us_avg = []
Us_mc = []
Us_cmp = []
# Us_bwq = []

# 蒙特卡洛仿真
for _ in range(loops_mc):
    for lamuda in lamudas:
        P_pre_mc = P0
        for _ in range(loops_kar):
            gamma = bernoulli.rvs(p=lamuda, size=1)
            inv_mc = np.matrix(C.dot(P_pre_mc).dot(C.transpose()) + R).I.item()
            P_mc = A.dot(P_pre_mc).dot(A.transpose()) + Q - \
                   gamma * inv_mc * A.dot(P_pre_mc).dot(C.transpose()).dot(C).dot(P_pre_mc).dot(A.transpose())
            P_pre_mc = P_mc
        B_mc = delt_e + P_mc.trace()
        U_mc = get_U(lamuda, B_mc)
        Us_mc.append(U_mc)

# 完全观测、不完全观测、确定性模型
for lamuda in lamudas:
    P_pre_avg = P0
    P_pre_cmp = P0
    # P_pre_bwq = P0
    for _ in range(loops_kar):
        # 确定性不完全观测模型
        inv_avg = np.matrix(C.dot(P_pre_avg).dot(C.transpose()) + R).I.item()
        P_avg = A.dot(P_pre_avg).dot(A.transpose()) + Q - \
                lamuda * inv_avg * A.dot(P_pre_avg).dot(C.transpose()).dot(C).dot(P_pre_avg).dot(A.transpose())
        P_pre_avg = P_avg

        # 不完全观测
        # inv_bwq = np.matrix(C.dot(P_pre_bwq).dot(C.transpose()) + R).I.item()
        # P_bwq = A.dot(P_pre_bwq).dot(A.transpose()) + Q - \
        #         lamuda * inv_bwq * A.dot(P_pre_bwq).dot(C.transpose()).dot(C).dot(P_pre_bwq).dot(A.transpose())
        # P_pre_bwq = P_bwq

        # 完全观测
        inv_cmp = np.matrix(C.dot(P_pre_cmp).dot(C.transpose()) + R).I.item()
        P_cmp = A.dot(P_pre_cmp).dot(A.transpose()) + Q - \
                inv_cmp * P_pre_cmp.dot(C.transpose()).dot(C).dot(P_pre_cmp)
        P_pre_cmp = P_cmp
    B_avg = delt_e + P_avg.trace()
    # B_bwq = delt_e + P_bwq.trace()
    B_cmp = delt_e + P_cmp.trace()
    U_avg = get_U(lamuda, B_avg)
    # U_bwq = get_U(lamuda, B_bwq)
    U_cmp = get_U(lamuda, B_cmp)
    Us_avg.append(U_avg)
    # Us_bwq.append(U_bwq)
    Us_cmp.append(U_cmp)

plt.rcParams['font.sans-serif'] = ['STZhongsong']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.grid(True)
plt.xlabel('λ')
plt.ylabel('U(bps)')
plt.plot(lamudas, Us_avg, label='不完全时钟同步模型')
plt.scatter([lamudas] * loops_mc, Us_mc, color='red', marker='.', alpha=0.9, label='蒙特卡洛随机实验')
plt.plot(lamudas, Us_cmp, label='完全时钟同步模型')
# plt.plot(lamudas, Us_bwq, label='不完全时钟同步模型')
plt.legend()
plt.savefig('fig5')
plt.show()
