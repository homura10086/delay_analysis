import numpy as np
from scipy.optimize import minimize
from v3_delay_analysis import get_dcp
from Lyapunov_v1 import get_step

P_max = 50  # dBm


def objective(Pt):
    lamuda, snr = get_dcp(Pt)
    _, _, target = get_step(lamuda, snr)
    return target


def constrain1(Pt):
    return -Pt


def constrain2(Pt):
    return Pt - P_max


P0 = 10
bnds = [(0, P_max)]
con1 = {"type": "ineq", "fun": constrain1}
con2 = {"type": "ineq", "fun": constrain2}
cons = ([con1, con2])
solution = minimize(objective, np.array(P0), bounds=bnds, method='SLSQP')
print(solution)
