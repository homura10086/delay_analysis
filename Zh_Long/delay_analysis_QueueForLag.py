"""
基于排队论的时延确定性分析，拉格朗日算法调用接口
"""

from math import log2, inf

from scipy import integrate
from scipy import stats

from delay_analysis_QueueAndSNCForLag import t_low, h, sigma, L, Bw, lamda_a, t_up


def get_dcp_queue(Pt: float, t_low=t_low, lamda_a=lamda_a, Bw=Bw) -> (float, float):
    Pt = pow(10, Pt / 10)  # mW
    snr = Pt * h  # 比值

    #   香农公式计算传输时延一阶矩
    r = lambda z: log2(1 + snr * z) * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)
    r_avg = integrate.quad(r, 0, inf)[0]
    x_avg = L / (Bw * r_avg)

    #   香农公式计算传输时延二阶矩
    x_square = \
        lambda z: (L ** 2 * stats.expon.pdf(z, loc=0, scale=2 * sigma ** 2)) / ((Bw * log2(1 + snr * z)) ** 2)
    x_square_avg = integrate.quad(x_square, 0, inf)[0]

    # 业务繁忙率
    rho = lamda_a * x_avg

    # 排队时延
    if rho < 1:
        W = (lamda_a * x_square_avg) / (2 * (1 - rho))  # 排队时延
    else:
        W = inf
    if W < 0:
        W = inf

    #   蒙特卡罗仿真dcp计算
    # dpc_mcs = []
    # for _ in range(loops):
    #     z_r = stats.expon.rvs(loc=0, scale=2 * sigma ** 2, size=times, random_state=None)  # 蒙特卡洛随机生成
    #     r_mc = Bw * np.log2(1 + snr * z_r)
    #     x_mc = L / r_mc
    #     rho_mc = lamda * x_mc
    #     w_mc = (lamda * x_square_avg) / (2 * (1 - rho_mc))
    #     w_mc[rho_mc >= 1] = inf
    #     w_mc[w_mc < 0] = inf
    #     w_mc_tmp = w_mc[w_mc <= t_up]
    #     dcp_mc = len(w_mc_tmp[w_mc_tmp >= t_low]) / len(w_mc)
    #     dpc_mcs.append(dcp_mc)

    #   DCP计算
    z_up = (pow(2, L / (Bw * (t_up - W))) - 1) / snr
    # 传输时延的下界大于0
    if t_low - W > 10e-06:
        # print('分母', t_low - W, 'snr', snr)
        z_low = (pow(2, L / (Bw * (t_low - W))) - 1) / snr
        p_upper = 1 - stats.expon.cdf(z_up, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
        p_lower = 1 - stats.expon.cdf(z_low, loc=0, scale=2 * sigma ** 2)  # 小于时延下界概率
    # 传输时延的下界小于0 且上界大于0
    elif t_low - W < 0 and t_up - W > 0:
        p_upper = 1 - stats.expon.cdf(z_up, loc=0, scale=2 * sigma ** 2)  # 小于时延上界概率
        p_lower = 0.0    # 小于时延下界概率
    # 传输时延上界小于0
    else:
        p_upper = p_lower = 0.0
    p_delay = p_upper - p_lower  # 时延落在上下界区间内的概率

    # print('传输时延 ', '%.3e' % x_avg * 1e3)
    # print('排队时延 ', '%.3e' % W * 1e3)
    # print('概率 ', p_delay)

    return p_delay, snr
