"""
拉格朗日-KKT理论推导 VS 算法求解
"""
from math import exp, log10
import matplotlib.pyplot as plt
import numpy as np
from delay_analysis_QueueAndSNCForLag import sigma, L, t_up_x, alpha, Bw, h, get_dcp_snc, t_low
from LyapunovSimple import A, C, R, q, get_step, delt_e, Ts, bk, v1, v2, tao, loops
from Lagrange_kkt_solve import get_solution
from tqdm import tqdm


# 求近似解
def get_pt(p_min: float, p_max: float, v1: float, v2: float, B=0.0, t_low=t_low) -> (float, float):
    Pts = np.arange(p_min, p_max + 1, 1)
    Us = []
    targets = []
    Pks = []
    for Pt in Pts:
        lamuda, snr = get_dcp_snc(Pt, t_low)
        Pk, U, target = get_step(lamuda, snr, loops, B, v1, v2)[:3]
        targets.append(target)
        Us.append(U)
        Pks.append(Pk)

    index = np.argmin(targets)
    return Pks[index], Us[index]


# 理论推导
def get_power(Pt, t_low):
    Pt = pow(10, Pt / 10)  # mW
    snr = Pt * h  # 比值
    z_up = (pow(2, L / (Bw * t_up_x)) - 1) / snr
    z_low = (pow(2, L / (Bw * t_low)) - 1) / snr
    lamuda = exp(- z_up / (2 * sigma**2)) - exp(- z_low / (2 * sigma**2))
    # lamuda, snr = get_dcp(Pt)
    Pk, _, target, a_avg = get_step(lamuda, snr)
    B = Pk + delt_e
    W = (Ts / (Ts + B)) * Bw

    # for test
    p_opt = (pow(2 + tao, (v1 * Pk * a_avg - target) / (v2 * W)) - 1) / h

    # diff_lamuda = exp(- z_up / (2 * sigma**2)) * z_up - exp(- z_low / (2 * sigma**2)) * z_low
    # diff_Pk = (Pk * A * C)**2 * diff_lamuda / ((2 * Pk * C**2 + R) * (A**2 - 1) + C**2 * (q - 2 * Pk * lamuda * A**2))
    # if Pk >= bk:
    #     p_opt = (pow(2, bk * diff_Pk * v1 / (v2 * W * h)) - 1) / h
    # else:
    #     p_opt = (pow(2, 2 * Pk * diff_Pk * v1 / (v2 * W * h)) - 1) / h
    return 10 * log10(p_opt)


if __name__ == '__main__':
    # 求得的最优解随时延下界的变化
    t_lows = np.arange(0.3e-3, 0.66e-3, 0.01e-3)
    pt_la_s = []
    pt_th_s = []
    solution_analysis_s = []
    with tqdm(total=len(t_lows)) as pbar:
        pbar.set_description('Processing')
        for t_low in t_lows:

            # SLSQP 算法求解
            pt_la = get_solution(t_low)
            # print(pt_la)
            pt_la_s.append(pt_la)

            # 近似求解
            pt_th = get_pt(t_low)
            pt_th_s.append(pt_th)

            # 理论推导
            solution_analysis = get_power(pt_th, t_low)    # 带入解析式求解
            # print(solution_analysis)
            solution_analysis_s.append(solution_analysis)
            # print(get_power())
            pbar.update(1)

    # print(pt_la_s)
    # print(solution_analysis_s)
    plt.plot(t_lows * 1e3, pt_la_s, label='Lagrange-KKT', marker='+')
    plt.plot(t_lows * 1e3, solution_analysis_s, label='Theory analysis', marker='.')
    plt.plot(t_lows * 1e3, pt_th_s, label='Approximate optimal', marker='^')
    plt.legend()
    plt.xlabel('delay lower bound (ms)')
    plt.ylabel('solution of power (dBm)')

    plt.grid(True)
    plt.show()
