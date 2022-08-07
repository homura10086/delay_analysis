"""
交替迭代求解保时带和功率的优化算法
"""
import copy
from typing import Callable
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from LyapunovSimple import get_step, get_stepForParam, get_karman, get_karmanForSlot, delt_e, P0, loops, get_U, v1, v2, Ps
from Zh_Long.Geat import geat
from delay_analysis_QueueAndSNCForLag import get_dcp_snc, lamda_a, h
from delay_analysis_QueueForLag import get_dcp_queue

p_max = 50.0  # dBm
p_min = 10.0  # dBm
p0 = (p_min + p_max) / 2
Pks = []
Uks = []


def objective(args: tuple, loops: int, modelName: str) -> Callable:
    """args = (B, V1, V2, lamda_a, Ps, a_s)"""
    B = args[0]
    V1 = args[1]
    V2 = args[2]
    lamda_a = args[3]
    Ps = args[4]
    a_s = args[5]
    if modelName == 'snc':
        return lambda x: get_step(
            Ps, a_s, get_dcp_snc(x, lamda_a=lamda_a)[0], pow(10, x / 10) * h, loops, B, V1, V2)[2]
    else:
        return lambda x: get_step(
            Ps, a_s, get_dcp_queue(x, lamda_a=lamda_a)[0], pow(10, x / 10) * h, loops, B, V1, V2)[2]


# SLSQP求解算法求解发射功率
def getSolutionPower(B: float, V1: float, V2: float, lamda_a: float, loops: int,
                     modeName: str, Ps: list, a_s: list) -> float:
    # 初值
    bounds = [(p_min, p_max)]
    # 参数
    args = (B, V1, V2, lamda_a, Ps, a_s)
    solution = minimize(
        objective(args, loops, modeName), np.array(p0), bounds=bounds, method='SLSQP')
    return solution.x.item()


# 保时带线性规划问题
def getSolutionBand(Pk: float) -> float:
    return Pk + delt_e


# 返回迭代算法最终求解的时钟同步精度和吞吐量
def getSolution(N: int, V1=v1, V2=v2, lamda_a=lamda_a, loops=loops, modelName='snc') -> (float, float):
    # N = 20  # 最大迭代次数
    delt_b = 0.0  # bk最大允许误差
    delt_p = 0.0  # pk最大允许误差
    B0 = P0 + delt_e
    pk = p0
    Bk = B0
    Bk_pre = 0.0
    pk_pre = 0.0
    Uk = 0.0
    Pk = 0.0
    k = 1
    # ps = [p0]
    # Bs = [B0]
    # print(p0, B0)
    # print("optimal: ", get_pt(p_min, p_max))

    Ps = [P0]
    a_s = []

    '交替迭代求解'
    with tqdm(total=N) as bar:
        bar.set_description('Optimizing')
        while k <= N and (abs(Bk - Bk_pre) >= delt_b or abs(pk - pk_pre) >= delt_p):
            '固定pk求解Bk'
            if modelName == 'snc':
                dcp = get_dcp_snc(pk, lamda_a=lamda_a)[0]
            elif modelName == 'queue':
                dcp = get_dcp_queue(pk, lamda_a=lamda_a)[0]
            else:
                dcp = 1.0
            Pk = get_karmanForSlot(dcp, loops, Ps, a_s)[0]
            Bk_pre = Bk
            Bk = getSolutionBand(Pk)

            '固定Bk求解pk'

            # Ps和a_s的副本，为了不影响算法迭代中的数据
            Ps_cp = copy.copy(Ps)
            a_s_cp = copy.copy(a_s)

            pk_pre = pk
            pk = getSolutionPower(Bk, V1, V2, lamda_a, loops, modelName, Ps_cp, a_s_cp)
            snr = pow(10, pk / 10) * h  # 比值
            Uk = get_U(Bk, snr)

            '每时隙的迭代数据'
            Pks.append(Pk)
            Uks.append(Uk)
            k += 1

            bar.update(1)

    return Uk, Pk
    # plt.figure(1)
    # plt.grid(True)
    # plt.plot(np.linspace(0, N, N + 1), Bs, label='TimeBand')
    # plt.legend()
    #
    # plt.figure(2)
    # plt.grid(True)
    # plt.plot(np.linspace(0, N, N + 1), ps, label='power')
    # plt.legend()
    #
    # plt.show()


def objectiveForParam(args: tuple, loops: int, modelName: str) -> Callable:
    """args = (V1, V2, lamda_a)"""
    V1 = args[0]
    V2 = args[1]
    lamda_a = args[2]
    if modelName == 'snc':
        return lambda x: get_stepForParam(
            get_dcp_snc(x, lamda_a=lamda_a)[0], pow(10, x / 10) * h, loops, v1=V1, v2=V2)[2]
    else:
        return lambda x: get_stepForParam(
            get_dcp_queue(x, lamda_a=lamda_a)[0], pow(10, x / 10) * h, loops, v1=V1, v2=V2)[2]


def getSolutionPowerForParam(V1: float, V2: float, lamda_a: float, loops: int, modeName: str) -> float:
    # 初值
    bounds = [(p_min, p_max)]
    # 参数
    args = (V1, V2, lamda_a)
    solution = minimize(
        objectiveForParam(args, loops, modeName), np.array(p0), bounds=bounds, method='SLSQP')
    return solution.x.item()


def getSolutionForParma(V1=v1, V2=v2, lamda_a=lamda_a, loops=loops, modelName='snc', algorithmName='lag') -> (
        float, float):

    # Ps = [P0]
    if algorithmName == 'lag':
        pk = getSolutionPowerForParam(V1, V2, lamda_a, loops, modelName)
    else:
        pk = geat(loops, lamda_a, v1, v2, modelName, p_min, p_max)
    # print(pk)
    # if modelName == 'snc':
    dcp, snr = get_dcp_snc(pk, lamda_a=lamda_a)
    # else:
    #     dcp, snr = get_dcp_queue(pk, lamda_a=lamda_a)
    Pk = get_karman(dcp, loops=100)[0]
    Bk = Pk + delt_e
    Uk = get_U(Bk, snr)

    return Uk, Pk

