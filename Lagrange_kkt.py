"""
增广拉格朗日-KKT求解优化问题
"""

import numpy as np
from scipy.optimize import minimize
from sympy import symbols, diff, solve
from v3_delay_analysis import get_dcp
from Lyapunov_v2 import get_step

P_max = 50  # dBm


def objective(Pt):
    lamuda, snr = get_dcp(Pt)
    _, _, target = get_step(lamuda, snr)
    return target


def constrain1(Pt):
    return Pt


def constrain2(Pt):
    return P_max - Pt


# SLSQP
P0 = 25
bnds = [(0, P_max)]
con1 = {"type": "ineq", "fun": constrain1}
con2 = {"type": "ineq", "fun": constrain2}
cons = [con1, con2]
solution = minimize(objective, np.array(P0), bounds=bnds, method='SLSQP')
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
