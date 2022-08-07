"""
增广拉格朗日-KKT求解优化问题
"""

import numpy as np
from scipy.optimize import minimize
from sympy import symbols, diff, solve
from delay_analysis_QueueAndSNCForLag import get_dcp_snc, t_low
from LyapunovSimple import get_step, loops

P_max = 50.0  # dBm
P_min = 10.0  # dBm


def objective(args):
    fun = lambda x: get_step(get_dcp_snc(x, args)[0], get_dcp_snc(x, args)[1], loops)[2]
    # lamuda, snr = get_dcp(Pt, t_low)
    # _, _, target, _ = get_step(lamuda, snr)
    # return target
    return fun


def constrain1(Pt):
    return Pt - P_min


def constrain2(Pt):
    return P_max - Pt


def get_solution(t_low=t_low):
    # SLSQP
    P0 = (P_max + P_min) / 2
    bnds = [(P_min, P_max)]
    con1 = {"type": "ineq", "fun": constrain1}
    con2 = {"type": "ineq", "fun": constrain2}
    cons = [con1, con2]
    args = t_low
    solution = minimize(objective(args), np.array(P0), bounds=bnds, method='SLSQP')
    return solution.x.item()
    # return solution


if __name__ == '__main__':
    solution = get_solution(t_low)
    print(solution)

# 拉格朗日-KKT
# lamuda, miu, pt = symbols('lamuda, miu, pt')
# f = objective(pt)
# L = objective(pt) - lamuda * pt + miu * (pt - P_max)
# df_pt = diff(L, pt)
# df_lamuda = diff(L, lamuda)
# df_miu = diff(L, miu)
# res = solve([df_pt, df_lamuda, df_miu], [pt, lamuda, miu])
# print(res)
