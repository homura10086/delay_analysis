import math
from math import *
import numpy as np
from scipy.stats import ncx2, chi2
import matplotlib.pyplot as plt
from pylab import *
# from sympy import *
from scipy import integrate

# 1 设置参数


# 传输信噪比
pb = 10  # 传输功率，42dBm (30dBm=1w, +3dBm 功率*2) （基站的发射功率在数十瓦量级）
M_b = N_b = 2  # 均匀矩形阵列（URAS）参数 基站 默认2
M_i = N_i = 8  # 均匀矩形阵列（URAS）参数 IRS 默认8
# list_ni = [4, 5, 6, 7, 8, 9, 10, 11, 12] # 作变量时的取值
K_2 = K_3 = 1  # 默认取值为1，LOS和non-LOS比例相同，K取值越大，LOS所占比例越大

A_1 = 2  # 路径损失系数（该系数越大，信号损失越大）BS-to-IRS
A_2 = 2.5  # 路径损失系数（该系数越大，信号损失越大 IRS-to-
A_3 = 2.5
A_4 = 1  # 路径损失系数（该系数越大，信号损失越大）user-to-user

L_b = [0, 0, 20]  # 基站、IRS、用户的位置 单位m，(使用 m，是因为参考论文中使用的m，并给出了响应的计算公式)
L_i = [100, 0, 20]  # 基站、IRS、用户的位置 单位m，
L_dt = [100, 20, 1]  # 基站、IRS、用户的位置 单位m，
L_dr = [100, 30, 1]  # 基站、IRS、用户的位置 单位m，

ratio = 0.3333  # URAs 元件之间距离和波长之间的比值 （小于等于1 / 2），用于计算URAs的响应，最优相移不涉及该参数，随机相移的计算会用到

# 离开角、到达角
AoA_h_1 = AoA_v_1 = math.pi / 6
AoD_h_1 = AoD_v_1 = math.pi / 4
AoD_h_2 = AoD_v_2 = math.pi / 3
AoD_h_3 = AoD_v_3 = math.pi / 8

# 能量捕获过程
E_ec = 0.8  # RF-to-DC 的转化效率，（0.3-0.8)  参考论文默认是0.3
# 后向散射过程
P_ec = 0.625  # 进行能量转换的功率占总功率的比例(0~1) power of energy conversion 0.625
CU_b = 8.9e-6  # circuit threshold of backscatter  后向散射模式的最低电路门限 单位：w。8.9e-6
eff_b = 0.5  # 后向散射天线系数
# eff_b = 1 # 后向散射天线系数，默认取值是0 还是 0.5？
# 主动传输过程
T_eh = 0.9999  # time of energy harvestHTT模式中，能量捕获的时间占比，在该时间内全部的功率都用来进行能量转换
CU_a = 113e-6  # circuit threshold of active transmission(HTT) 主动传输模式的最低电路门限 单位：w， 113e-6
R_b = 2.057  # 后向散射覆盖中断概率 单位 bit/s/Hz，对应SNR门限5dB, 参考论文中只要高于阈值，就以预设的速率接收，预设速率是 1 kbps
R_a = 7.213e-05  # 主动传输中断概率 单位 bit/s/Hz， 对应信噪比门限-40dB
# 噪声
p_n = 9.99e-12  # 噪声功率 10pw = 10e-12w 单位:w (=-80dBm)【参考论文给的 -120 dBm/Hz 10e16W】
B = 1e6  # 单位：Hz ，考虑单位带宽的情况，就不需要此项


## URAs每个elements的响应函数--随机相移计算需要


#  URA中每个elements的响应函数： Input：水平角、垂直角、element所在行、所在列、element间隔与信号波长的比值
def URAresponse(a1, a2, m, n, r):
    return 2 * math.pi * r * math.sin(a2) * ((m - 1) * math.cos(a1) + (n - 1) * math.sin(a1))


## 大尺度信道增益计算


# 信道增益：来自参考论文，1/(1000 * (d ^ a) ) 计算结果随着距离的增加而减少， 输入参数分别为：路径损耗指数、位置1、位置2

def pathgain(a, d1, d2):
    return pow(np.sqrt(np.square(d1[0] - d2[0]) + np.square(d1[1] - d2[1]) + np.square(d1[2] - d2[2])), -a)


PL_1 = pathgain(A_1, L_b, L_i)
PL_2 = pathgain(A_2, L_i, L_dr)
PL_3 = pathgain(A_3, L_i, L_dt)
PL_4 = pathgain(A_4, L_dt, L_dr)
# print(PL_1, PL_2, PL_3, PL_4)

c11 = sqrt(PL_1)
c21 = sqrt(PL_2 * K_2 / (K_2 + 1))
c22 = sqrt(PL_2 * 1 / (K_2 + 1))
c31 = sqrt(PL_3 * K_3 / (K_3 + 1))
c32 = sqrt(PL_3 * 1 / (K_3 + 1))
c41 = sqrt(PL_4)
# print(c11, c21, c22, c31, c32, c41)

b1 = sqrt(eff_b * (1 - P_ec))

xi_1 = c22 * c21 * pow(c11, 2)
xi_2 = b1 * c41 * c31 * c22 * pow(c11, 2)
xi_3 = b1 * c41 * c32 * c21 * pow(c11, 2)
xi_4 = pow(b1, 2) * pow(c41, 2) * c32 * c31 * pow(c11, 2)
# print('\n', xi_1, '\n', xi_2, '\n', xi_3, '\n', xi_4)


## 最优相移情况下，输入参数计算


def opt_phase(M, N, h1, v1, h2, v2):
    # print('start:', ratio)
    sum_x = 0
    sum_y = 0
    for m in range(M):
        for n in range(N):
            phase_1 = URAresponse(h1, v1, m + 1, n + 1, ratio) - URAresponse(h2, v2, m + 1, n + 1,
                                                                             ratio)  # 一个行向量 * 一个列向量 对应位置元素的乘积
            sum_x = sum_x + math.cos(phase_1)  # 实部
            sum_y = sum_y + math.sin(phase_1)  # 虚部
    # print('end:', ratio)
    return sum_x, sum_y, pow(sum_x, 2) + pow(sum_y, 2)  # 返回值1:实部，返回值2:模的平方，替代最优相移的 M_i * N_i * M_i * N_i=4096


# 31最优  中间变量表达式：一个实部Re{h2h3}，一个平方||h2øh1||^2
re_b31 = opt_phase(M_i, N_i, AoD_h_2, AoD_v_2, AoD_h_3, AoD_v_3)  # 取re_b31[0]
square_b2131 = M_b * N_b * re_b31[2]  ##计算时，应该时 3-2，但是由于取平方，所以2-3的结果是一样的
# print('re_b31, square_b2131 =', re_b31, square_b2131)
# 21最优  一个实部，一个平方
re_b21 = opt_phase(M_i, N_i, AoD_h_2, AoD_v_2, AoD_h_3, AoD_v_3)  # 取re_b31[0]
temp = opt_phase(M_i, N_i, AoD_h_3, AoD_v_3, AoD_h_2, AoD_v_2)  # 取re_b31[0]
square_b3121 = M_b * N_b * temp[2]
# print('re_b21, square_b3121 =', re_b21, square_b3121)

# B-mode 31最优
pl_b31 = pow(c21 * c11, 2) * square_b2131 + 2 * b1 * c41 * c31 * c21 * pow(c11, 2) * re_b31[0] + pow(b1, 2) * pow(c41,
                                                                                                                  2) * pow(
    c31, 2) * pow(c11, 2) * (M_b * N_b * M_i * N_i * M_i * N_i)

x1_square31 = pow(xi_1,
                  2) * M_b * N_b * M_i * N_i * square_b2131 + 2 * xi_1 * xi_2 * M_b * N_b * M_i * N_i * M_i * N_i * \
              re_b31[0] + pow(xi_2, 2) * M_b * N_b * M_i * N_i * M_b * N_b * M_i * N_i * M_i * N_i
x2_square31 = pow(xi_3,
                  2) * M_b * N_b * M_i * N_i * square_b2131 + 2 * xi_3 * xi_4 * M_b * N_b * M_i * N_i * M_i * N_i * \
              re_b31[0] + pow(xi_4, 2) * M_b * N_b * M_i * N_i * M_b * N_b * M_i * N_i * M_i * N_i

pnl_b31 = (x1_square31 + x2_square31) / pl_b31
# print('pl_b31, pnl_b31 =', pl_b31, pnl_b31)
##########
##########
# 异常值 pl偏大，pnl偏小
# 可能是pl的表达式有问题，特别是h4处理，检查一遍 2021-06-11 20:12

# B-mode 21最优
pl_b21 = pow(c21 * c11, 2) * (M_b * N_b * M_i * N_i * M_i * N_i) + 2 * b1 * c31 * pow(c21 * c11, 2) * re_b31[
    0] + b1 * b1 * c31 * c11 * c11 * square_b3121

x1_square21 = pow(xi_1,
                  2) * M_b * N_b * M_i * N_i * M_i * N_i * M_b * N_b * M_i * N_i * M_i * N_i + 2 * xi_1 * xi_2 * M_b * N_b * M_i * N_i * M_i * N_i * \
              re_b31[0] + xi_2 * xi_2 * M_b * N_b * square_b3121
x2_square21 = pow(xi_3,
                  2) * M_b * N_b * M_i * N_i * M_b * N_b * M_i * N_i * M_i * N_i + 2 * xi_3 * xi_4 * M_b * N_b * M_i * N_i * M_i * N_i * \
              re_b31[0] + xi_4 * xi_4 * M_b * N_b * M_i * N_i * square_b3121

pnl_b21 = (x1_square21 + x2_square21) / pl_b21
# print('pl_b21, pnl_b21 =', pl_b21, pnl_b21)
# H-mode
pl_h31 = pow(c31, 2) * pow(c11, 2) * M_b * N_b * M_i * N_i * M_i * N_i
pnl_h31 = pow(c31, 2) * pow(c11, 2) * M_b * N_b * M_i * N_i
# print('pl_h31, pnl_h31 =', pl_h31, pnl_h31)
pl_h21 = pow(c21 * c11, 2) * M_b * N_b * M_i * N_i * M_i * N_i
pnl_h21 = pow(c21 * c11, 2) * M_b * N_b * M_i * N_i
# print('pl_h21, pnl_h21 =', pl_h21, pnl_h21)


###### P_l, P_nl =  0.00045794672179195697 7.155417527999328e-06

## 随机相移响应计算

# Random phase shifts 随机相移， INPUT：URA行数、URA列数、水平角1、垂直角1、水平角2、垂直角2；
# OUTPUT：多个复数和的模方
# 最优相移相移是一个实数，（即一个复数的模方），随机相移的响应也是如此
def random_phase(M, N, h1, v1, h2, v2):
    sum_x = 0
    sum_y = 0
    np.random.seed(3)
    for m in range(M):
        for n in range(N):
            phase_0 = np.random.uniform(0, 2 * math.pi)  # 随机相移
            phase_1 = URAresponse(h1, v1, m + 1, n + 1, ratio) - URAresponse(h2, v2, m + 1, n + 1, ratio) + phase_0
            # phase_1 = 0 相移后的相位相同时，和取到最值 4096
            sum_x = sum_x + math.cos(phase_1)  # 实部
            sum_y = sum_y + math.sin(phase_1)  # 虚部
    # print(pow(sum_x, 2) + pow(sum_y, 2))
    return pow(sum_x, 2) + pow(sum_y, 2)  # 模的平方，替代最优相移的 M_i * N_i * M_i * N_i


## 随机相移情况下，输入参数计算


# B-mode 一种情况
temp21 = random_phase(M_i, N_i, AoD_h_2, AoD_v_2, AoA_h_1, AoA_v_1)
temp31 = random_phase(M_i, N_i, AoD_h_3, AoD_v_3, AoA_h_1, AoA_v_1)
# print(temp31, temp21)
#### 120.36506854371032

pl_b_ran = pow(c21 * c11, 2) * (M_b * N_b * temp21) + 2 * b1 * c41 * c31 * c21 * pow(c11, 2) * re_b31[0] + pow(b1,
                                                                                                               2) * pow(
    c41, 2) * pow(c31, 2) * pow(c11, 2) * (M_b * N_b * temp31)

x1_square31_ran = pow(xi_1, 2) * M_b * N_b * M_i * N_i * (
        M_b * N_b * temp21) + 2 * xi_1 * xi_2 * M_b * N_b * M_i * N_i * M_i * N_i * re_b31[
                      0] + xi_2 * xi_2 * M_b * N_b * M_i * N_i * (M_b * N_b * temp31)
x2_square31_ran = pow(xi_3, 2) * M_b * N_b * M_i * N_i * (
        M_b * N_b * temp21) + 2 * xi_3 * xi_4 * M_b * N_b * M_i * N_i * M_i * N_i * re_b31[
                      0] + xi_4 * xi_4 * M_b * N_b * M_i * N_i * (M_b * N_b * temp31)

pnl_b_ran = (x1_square31_ran + x2_square31_ran) / pl_b_ran
# print('pl_b_ran, pnl_b_ran =', pl_b_ran, pnl_b_ran)
############################################
############################################
# 数量级相差太多，pl_ran 偏大，pnl_ran偏小，表达式中，pnl受pl影响，随pl增大而减小

# H-mode 分情况
# 31
pl_h31_ran = pow(c31 * c11, 2) * M_b * N_b * temp31
pnl_h31_ran = pow(c31 * c11, 2) * M_b * N_b * M_i * N_i
# print('pl_h31_ran, pnl_h31_ran =', pl_h31_ran, pnl_h31_ran)
# 21
pl_h21_ran = pow(c21 * c11, 2) * M_b * N_b * temp21
pnl_h21_ran = pow(c21 * c11, 2) * M_b * N_b * M_i * N_i
# print('pl_h21_ran, pnl_h21_ran =', pl_h21_ran, pnl_h21_ran)

##### P_l_random, P_nl_random= 1.3457223769007896e-05 7.155417527999328e-06

## 非中心卡方分布输入参数计算
### 自由度(degrees of freedom) 非中心参数(non-centrality parameter) 阈值(threshold)

df = 2  # 都是2


def NC(pl, pn):  # 非中心参数
    return 2 * pl / pn


# B-mode 31最优
nc_b31 = NC(pl_b31, pnl_b31)
nc_b31_ran = NC(pl_b_ran, pnl_b_ran)
# print('nc_b31, nc_b31_ran =', nc_b31, nc_b31_ran)

# B-mode 21最优
nc_b21 = NC(pl_b21, pnl_b21)
nc_b21_ran = NC(pl_b_ran, pnl_b_ran)
# print('nc_b21, nc_b21_ran =', nc_b21, nc_b21_ran)

# H-mode 31
nc_h31 = NC(pl_h31, pnl_h31)
nc_h31_ran = NC(pl_h31_ran, pnl_h31_ran)
# print('nc_h31, nc_h31_ran =', nc_h31, nc_h31_ran)

# H-mode 21
nc_h21 = NC(pl_h21, pnl_h21)
nc_h21_ran = NC(pl_h21_ran, pnl_h21_ran)
# print('nc_h21, nc_h21_ran =', nc_h21, nc_h21_ran)

########### nc, nc_random = 128.0 3.7614083919909476

# print(r_mode, c_b, c_h31, c_h21)

######## 画一下，给定 非中心参数 NC 的概率分布值
########   NC 越大，其每个取值点的概率密度越小
# nc_test = nc_b31
nc_test = nc_b21
nc_test = nc_h31
# nc_test = nc_h21
fig, ax = plt.subplots(1, 1)
x = np.linspace(ncx2.ppf(0.01, df, nc_test), ncx2.ppf(0.99, df, nc_test), 100)
ax.plot(x, ncx2.pdf(x, df, nc_test), 'r-', lw=5, alpha=0.6, label='ncx2 pdf')
# ax.semilogy(x, ncx2.pdf(x, df, nc_test), 'r-', lw=5, alpha=0.6, label='ncx2 pdf')
# ax.plot(x, ncx2.cdf(x, df, nc_test, loc=0, scale=1), 'b-', lw=5, alpha=0.6, label='ncx2 cdf - optimal phase shift')
plt.xlabel('threshold')
plt.ylabel('probability')
ax.legend(loc='best', frameon=False)
# print(ncx2.ppf(0, df, nc_b31), ncx2.ppf(1, df, nc_b31))

# fig, ax = plt.subplots(1, 1)
# x = np.linspace(ncx2.ppf(0.01, df, nc_b31_ran), ncx2.ppf(0.99, df, nc_b31_ran), 100)
# ax.plot(x, ncx2.pdf(x, df, nc_b31_ran), 'r-', lw=5, alpha=0.6, label='ncx2 pdf')
# # ax.plot(x, ncx2.cdf(x, df, nc_b31_ran, loc=0, scale=1), 'b-', lw=5, alpha=0.6, label='ncx2 cdf - random phase shift')
# plt.xlabel('threshold')
# plt.ylabel('probability')
# ax.legend(loc='best', frameon=False)

## 覆盖概率 VS 基站发射功率

# 数组存储覆盖概率
c_prob_array = []
c_prob_random_array = []
c_prob_pure_a_array = []
c_prob_pure_b_array = []
c_prob_r_array = []
# 数组存储吞吐量
v_overall_array = []
v_overall_random_array = []
v_b_pure_array = []
v_a_pure_array = []
v_r_array = []
# 数组存储基站发射功率
pb_array = []

# 接收速率阈值
R_b = 2.057
R_a = 7.213e-05

CU_b = 8.9e-6
CU_a = 113e-6

# 数组存储选择 backscatter 模式的概率
prob_mode_array = []
prob_mode_random_array = []

# 初始功率
# pb = 0.05
pb = 0.00001


# 阈值threshold计算函数，输入参数：临界条件r，参数P_nl，输出：阈值
def threshold(r, pnl):
    return 2 * r / pnl


r_mode_threshold_array = []
c_b31_threshold_array = []
c_h31_threshold_array = []
c_h21_threshold_array = []

for i in range(10):
    # 计算临界条件
    r_mode = CU_a / (T_eh * E_ec * pb)  # 模式选择
    c_b = ((pow(2, R_b) - 1) * p_n) / pb  # 后向散射覆盖阈值
    c_h31 = ((pow(2, R_a) - 1) * (1 - T_eh) * p_n + CU_a * PL_4) / (T_eh * E_ec * pb * PL_4)  # HTT覆盖阈值
    c_h21 = ((pow(2, R_a) - 1) * p_n) / pb  # 没有DT 覆盖阈值
    # 计算阈值，并放在一个数组里
    r_mode_threshold_array.append(threshold(r_mode, pnl_h31))
    c_b31_threshold_array.append(threshold(c_b, pnl_b31))
    c_h31_threshold_array.append(threshold(c_h31, pnl_h31))
    c_h21_threshold_array.append(threshold(c_h21, pnl_h21))

    # print('r_mode_threshold:', threshold(r_mode, pnl_h31))
    # print('c_b_threshold_31:', threshold(c_b, pnl_b31))
    # print('c_b_threshold_21:', threshold(c_b, pnl_b21))
    # print('c_h31_threshold:', threshold(c_h31, pnl_h31))
    # print('c_h21_threshold:', threshold(c_h21, pnl_h21))

    # backscatter 模式覆盖概率
    c_b_prob = 1 - ncx2.cdf(threshold(c_b, pnl_b31), df, nc_b31, loc=0, scale=1)  # 最优相移中断概率
    c_b_prob_random = 1 - ncx2.cdf(threshold(c_b, pnl_b_ran), df, nc_b31_ran, loc=0, scale=1)

    # HTT 模式覆盖概率
    c_a_prob = 1 - ncx2.cdf(threshold(c_h31, pnl_h31), df, nc_h31, loc=0, scale=1)  # 最优相移中断概率
    c_a_prob_random = 1 - ncx2.cdf(threshold(c_h31, pnl_h31_ran), df, nc_h31_ran, loc=0, scale=1)
    # 没有 DT 的覆盖概率，直接到接收端 r
    c_r_prob = 1 - ncx2.cdf(threshold(c_h21, pnl_h21), df, nc_h21, loc=0, scale=1)

    # backscatter 模式选择概率
    prob_mode = ncx2.cdf(threshold(r_mode, pnl_h31), df, nc_h31, loc=0, scale=1)
    prob_mode_random = ncx2.cdf(threshold(r_mode, pnl_h31_ran), df, nc_h31_ran, loc=0,
                                scale=1)  ## 按照随机参数计算选择概率时，选B-mode的概率几乎一直为1，所以这里选择概率 是不是 按照 最优的 概率来选
    # prob_mode_random = prob_mode ## 选择概率 按照 最优的 概率来选, 出现的问题：随机相移下 HTT的覆盖概率几乎一直为0，导致后面下降，所以不能改
    prob_mode_array.append(prob_mode)  # 模式选择概率存储在数组中
    prob_mode_random_array.append(prob_mode_random)
    # 最优
    c_prob = prob_mode * c_b_prob + (1 - prob_mode) * c_a_prob  # 覆盖概率
    c_prob_array.append(c_prob)

    if prob_mode > 1:  # 防止概率超过1，导致后面的计算出现负值
        prob_mode = 1
    # 随机
    c_prob_random = prob_mode_random * c_b_prob_random + (1 - prob_mode_random) * c_a_prob_random
    c_prob_random_array.append(c_prob_random)
    # print("xm, xcb, xca = ", x_m, x_c_b, x_c_a)
    # print("mode, c_b, c_a, c, mode_radom = ", prob_mode, c_b_prob, c_a_prob, c_prob, prob_mode_random)
    # 纯后向散射 和 纯HHT 模式下的 覆盖概率
    c_prob_pure_a = c_a_prob
    c_prob_pure_a_array.append(c_prob_pure_a)

    c_prob_pure_b = c_b_prob
    c_prob_pure_b_array.append(c_prob_pure_b)

    # 没有DT的覆盖概率
    c_prob_r_array.append(c_r_prob)

    pb_array.append(pb)  # 基站发射功率
    # pb = pb + 0.001
    pb = pb + 0.5

# print('r_mode_threshold_array = \n', r_mode_threshold_array)
# print('prob_mode_array = \n', prob_mode_array)
# print('prob_mode_random_array = \n', prob_mode_random_array)
plt.cla()  # 清空旧图
# figure(1)
# # figure(1, dpi=300)
# # plt.title('Coverage probability versus Transmit Power')
# plt.plot(pb_array, c_prob_pure_b_array, color='#00CC00', linestyle='-', marker='+', markerfacecolor='none',
#          markeredgewidth=1, lw=2, alpha=0.9, label='Pure backscatter')
# plt.plot(pb_array, c_prob_pure_a_array, color='#123EAB', linestyle='-', marker='v', markerfacecolor='none',
#          markeredgewidth=1, lw=2, alpha=0.9, label='Pure HTT')
# # plt.plot(pb_array, c_prob_random_array, 'r-', marker='s', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Random Phase Shift')
# plt.plot(pb_array, c_prob_r_array, color='#FFAB00', linestyle='-', marker='o', markerfacecolor='none',
#          markeredgewidth=1, lw=2, alpha=0.9, label='Without DT')
# plt.plot(pb_array, c_prob_array, color='#FF0000', linestyle='-', marker='o', markerfacecolor='none', markeredgewidth=1,
#          lw=2, alpha=0.9, label='The proposed')

# print('c_prob_pure_b_array = \n', c_prob_pure_b_array)
# print('c_prob_pure_a_array = \n', c_prob_pure_a_array)
# print('c_prob_random_array = \n', c_prob_random_array)
# print('c_prob_array = \n', c_prob_array)
# print('c_prob_r_array = \n', c_prob_r_array)

plt.grid(True)
# plt.xlim(0.03, 0.12)
# plt.ylim(0, 1)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel('Transmit Power', fontsize=12)
plt.ylabel('Coverage Probability', fontsize=12)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.3), fancybox=True, shadow=False, frameon=True, fontsize=12)

# figure(2)
# figure(1, dpi=300)

# semilogy(pb_array, c_prob_pure_b_array, 'g-', marker='+', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Pure Backscatter')
# semilogy(pb_array, c_prob_pure_a_array, 'y-', marker='v', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Pure HTT')
# semilogy(pb_array, c_prob_random_array, 'r-', marker='s', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Random Phase Shift')
# semilogy(pb_array, c_prob_array, 'b-', marker='o', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Optimal Phase Shift')

# plt.grid(True)
# plt.xlim(1, 8)
# plt.ylim(0.1, 1.02)
# plt.tick_params(axis='both',which='major',labelsize=12)
# plt.xlabel('Transmit Power', fontsize=12)
# plt.ylabel('Coverage Probability', fontsize=12)
# plt.legend(loc='center right', bbox_to_anchor=(1, 0.3), fancybox=True, shadow=False, frameon=True, fontsize=12)


# 吞吐量 VS 基站发射功率


# 传输信噪比
# pbsnr = 140 # 默认取值
# P_b_SNR = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150] # 作变量时的取值
# pb = 16 # 传输功率，42dBm (30dBm=1w, +3dBm 功率*2) （基站的发射功率在数十瓦量级）
# M_b = N_b = 2 # 均匀矩形阵列（URAS）参数 基站 默认2
# M_i = N_i = 8 # 均匀矩形阵列（URAS）参数 IRS 默认8
# list_ni = [4, 5, 6, 7, 8, 9, 10, 11, 12] # 作变量时的取值
# K_iu = 1 # 默认取值 # 莱斯信道参数
# list_kiu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # 作变量时的取值
# A_bi = 2 # 路径损失系数（该系数越大，信号损失越大）BS-to-IRS
# A_iu = 2.5 # 路径损失系数（该系数越大，信号损失越大 IRS-to-user
# A_uu = 2 # 路径损失系数（该系数越大，信号损失越大）user-to-user
# L_b = [0, 0] # 基站、IRS、用户的位置 单位m，(使用 m，是因为参考论文中使用的m，并给出了响应的计算公式)
# L_i = [100, 0] # 基站、IRS、用户的位置 单位m，
# L_u1 = [100, 20] # 基站、IRS、用户的位置 单位m，
# L_u2 = [100, 30] # 基站、IRS、用户的位置 单位m，
# ratio = 0.3333  # URAs 元件之间距离和波长之间的比值 （小于等于1 / 2），用于计算URAs的响应，最优相移不涉及该参数，随机相移的计算会用到
# # 离开角、到达角
# AoA_h_bi = AoA_v_bi = math.pi / 6
# AoD_h_bi = AoD_v_bi = math.pi / 4
# AoD_h_iu = AoD_v_iu = math.pi / 3
# # 能量捕获过程
# E_ec = 0.5 # RF-to-DC 的转化效率，（0.3-0.8)  参考论文默认是0.3
# # 后向散射过程
# P_ec = 0.625 # 进行能量转换的功率占总功率的比例(0~1) power of energy conversion 0.625
# CU_b = 8.9e-6 # circuit threshold of backscatter  后向散射模式的最低电路门限 单位：w。8.9e-6
# eff_b = 1 # 后向散射天线系数
# R_b = 2.057 # outage rate of backscatter ， 后向散射中断概率 单位 bit/s/Hz，对应信载比门限5dB, 参考论文中只要高于阈值，就以预设的速率接收，预设速率是 1 kbps
# # 主动传输过程
# T_eh = 0.5 # time of energy harvestHTT模式中，能量捕获的时间占比，在该时间内全部的功率都用来进行能量转换
# CU_a = 113e-6 # circuit threshold of active transmission(HTT) 主动传输模式的最低电路门限 单位：w， 113e-6
# R_a = 7.213e-05 #4 outage rate of active transmission ， 主动传输中断概率 单位 bit/s/Hz， 对应信噪比门限-40dB
# # 噪声
# p_n = 9.99e-12 # power of noise 单位； w (=-80dBm)
# B = 1e6 # bandwidth 单位： Hz ，考虑单位带宽的情况，就不需要此项
##########################################################################################################################################################################################################################################################################################################
# 数组存储覆盖概率
c_prob_array = []
c_prob_random_array = []
c_prob_pure_a_array = []
c_prob_pure_b_array = []
# 数组存储吞吐量（4种方案：提出的最优方案、随机相移方案、纯后向散射、纯HTT）
v_overall_array = []
v_overall_random_array = []
v_b_pure_array = []
v_a_pure_array = []
v_r_array = []

# 数组存储基站发射功率
pb_array = []
# 初始功率
# pb = 0.02

# 接收速率阈值
# R_b = 20 #
# R_a = 20
R_b = 2.057 * 1e-12
R_a = 7.213e-05 * 3e5

# 电路启动阈值
CU_b = 8.9e-6
CU_a = 113e-6
# CU_b = 8.9e-4
# CU_a = 113e-5

# 初始功率
# pb = 0.04
pb = 0.00001

r_mode_threshold_array = []
c_b31_threshold_array = []
c_h31_threshold_array = []
c_h21_threshold_array = []
# 数组存储选择 backscatter 模式的概率
prob_mode_array = []
prob_mode_random_array = []

# pb = 8 # 在基站发射功率在 1 和 8 的情况下对比 后向散射天线效率的影响
# eff_b = 0.1

for i in range(16):
    r_mode = CU_a / (T_eh * E_ec * pb)
    c_b = ((pow(2, R_b) - 1) * p_n) / pb
    c_h31 = ((pow(2, R_a) - 1) * (1 - T_eh) * p_n + CU_a * PL_4) / (T_eh * E_ec * pb * PL_4)
    c_h21 = ((pow(2, R_a) - 1) * p_n) / pb
    ### 吞吐量 backscatter
    f = lambda x: log2(1 + pb * x / p_n) * ncx2.pdf(x, df, nc_b31)  # 平均吞吐量 最优相移 （使用积分计算）
    v_b_temp_0 = integrate.quad(f, c_b, np.inf)
    v_b = v_b_temp_0[0]
    ### 吞吐量 HTT
    g = lambda y: log2(1 + ((T_eh * E_ec * pb * y - CU_a) * PL_4) / ((1 - T_eh) * p_n)) * ncx2.pdf(y, df, nc_b31)
    v_a_temp_0 = integrate.quad(g, c_h31, np.inf)
    v_a = v_a_temp_0[0]
    ### 吞吐量 没有DT
    h = lambda z: log2(1 + pb * z / p_n) * ncx2.pdf(z, df, nc_h21)
    v_r_temp_0 = integrate.quad(h, c_h21, np.inf)
    v_r = v_r_temp_0[0]
    v_r_array.append(v_r)

    # backscatter 模式选择概率
    prob_mode = ncx2.cdf(threshold(r_mode, pnl_h31), df, nc_h31, loc=0, scale=1)
    prob_mode_random = ncx2.cdf(threshold(r_mode, pnl_h31_ran), df, nc_h31_ran, loc=0,
                                scale=1)  ## 按照随机参数计算选择概率时，选B-mode的概率几乎一直为1，所以这里选择概率 是不是 按照 最优的 概率来选
    # prob_mode_random = prob_mode ## 选择概率 按照 最优的 概率来选, 出现的问题：随机相移下 HTT的覆盖概率几乎一直为0，导致后面下降，所以不能改
    prob_mode_array.append(prob_mode)  # 模式选择概率存储在数组中
    prob_mode_random_array.append(prob_mode_random)

    v_overall = prob_mode * v_b + (1 - prob_mode) * v_a
    v_overall_array.append(v_overall)

    f_random = lambda x: log2(1 + 1 + pb * x / p_n) * ncx2.pdf(x, df, nc_b31_ran)  # 平均吞吐量 随机相移
    v_b_temp_0_random = integrate.quad(f_random, c_b, np.inf)
    v_b_random = v_b_temp_0_random[0]

    g_random = lambda y: log2(1 + ((T_eh * E_ec * pb * y - CU_a) * PL_4) / ((1 - T_eh) * p_n)) * ncx2.pdf(y, df,
                                                                                                          nc_b31_ran)
    v_a_temp_0_random = integrate.quad(g_random, c_h31, np.inf)
    v_a_random = v_a_temp_0_random[0]

    v_overall_random = prob_mode_random * v_b_random + (1 - prob_mode_random) * v_a_random
    v_overall_random_array.append(v_overall_random)

    v_b_pure = v_b  # 平均吞吐量 单一模式
    v_b_pure_array.append(v_b_pure)
    v_a_pure = v_a
    v_a_pure_array.append(v_a_pure)

    # pb_array.append(eff_b)    # 基站发射功率
    pb_array.append(pb)  # 基站发射功率
    # eff_b = eff_b + 0.05
    pb = pb + 0.4

# print('r_mode_threshold_array = \n', r_mode_threshold_array)
# print('prob_mode_array = \n', prob_mode_array)
# print('prob_mode_random_array = \n', prob_mode_random_array)

# print('pb_array = ', pb_array)
# print('v_b_pure_array = ', v_b_pure_array)
# print('v_a_pure_array = ', v_a_pure_array)
# print('v_overall_random_array = ', v_overall_random_array)
# print('v_overall_array = ', v_overall_array)

# figure(2, dpi=300)
figure(1)
# plt.title('Average Throughput versus Backscattering Efficiency')
plt.plot(pb_array, v_b_pure_array, color='#00CC00', linestyle='-', marker='+', markerfacecolor='none',
         markeredgewidth=1, lw=2, alpha=0.9, label='Pure backscatter')
plt.plot(pb_array, v_a_pure_array, color='#123EAB', linestyle='-', marker='v', markerfacecolor='none',
         markeredgewidth=1, lw=2, alpha=0.9, label='Pure HTT')
# plt.plot(pb_array, v_overall_random_array, 'r-', marker='o', markerfacecolor='none', markeredgewidth=1, lw=2, alpha=0.9, label='Random Phase Shift')
plt.plot(pb_array, v_r_array, color='#FFAB00', linestyle='-', marker='o', markerfacecolor='none', markeredgewidth=1,
         lw=2, alpha=0.9, label='Without DT')
plt.plot(pb_array, v_overall_array, color='#FF0000', linestyle='-', marker='o', markerfacecolor='none',
         markeredgewidth=1, lw=2, alpha=0.9, label='The proposed')

plt.grid(True)
# plt.xlim(0.1, 0.85)
# plt.ylim(2, 18) # 限定纵轴范围
plt.tick_params(axis='both', which='major', labelsize=12)

# plt.xlabel('Backscattering Efficiency',fontsize=12)
plt.xlabel('Transmit Power', fontsize=12)
plt.ylabel('Average Throughput (bit/s/Hz)', fontsize=12)
plt.legend(loc='best', bbox_to_anchor=(0.5, 0.5), fancybox=True, shadow=False, frameon=True, fontsize=12)
plt.show()
# figure(3)
# plt.title('The Selection Probability of Backscatter Mode')
# plt.plot(pb_array, prob_mode_array, 'b')
# plt.plot(pb_array, prob_mode_random_array, 'r')
