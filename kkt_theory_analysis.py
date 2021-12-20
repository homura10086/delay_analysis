"""
拉格朗日-KKT理论推导
"""
from math import exp, log10
from v3_delay_analysis import sigma, L, t_up_x, t_low, alpha, Bw, h, get_dcp
from Lyapunov_v2 import A, C, R, q, get_step, delt_e, Ts, bk, v1, v2


def get_power(Pt):
    Pt = pow(10, Pt / 10)  # mW
    snr = Pt * h  # 比值
    z_up = (pow(2, L / (Bw * t_up_x)) - 1) / snr
    z_low = (pow(2, L / (Bw * t_low)) - 1) / snr
    lamuda = exp(- z_up / (2 * sigma**2)) - exp(- z_low / (2 * sigma**2))
    # lamuda, snr = get_dcp(Pt)
    Pk, _, _ = get_step(lamuda, snr)
    B = Pk + delt_e
    W = (Ts / (Ts + B)) * Bw
    diff_lamuda = exp(- z_up / (2 * sigma**2)) * z_up - exp(- z_low / (2 * sigma**2)) * z_low
    diff_Pk = (Pk * A * C)**2 * diff_lamuda / ((2 * Pk * C**2 + R) * (A**2 - 1) + C**2 * (q - 2 * Pk * lamuda * A**2))
    if Pk >= bk:
        p_opt = (pow(2, bk * diff_Pk * v1 / (v2 * W * h)) - 1) / h
    else:
        p_opt = (pow(2, 2 * Pk * diff_Pk * v1 / (v2 * W * h)) - 1) / h
    # return 10 * log10(p_opt)
    return p_opt


print(get_power(32))
