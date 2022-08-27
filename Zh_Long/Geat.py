# -*- coding: utf-8 -*-

import geatpy as ea

from Zh_Long.LyapunovSimple import get_stepForParam, v1, v2
from Zh_Long.delay_analysis_QueueAndSNCForLag import get_dcp_snc, h, lamda_a
from Zh_Long.delay_analysis_QueueForLag import get_dcp_queue


class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, loops: int, pMin: float, pMax: float, Bw: float, lamda_a: float, v1: float, v2: float,
                 modelName: str):
        self.loops = loops
        self.v2 = v2
        self.v1 = v1
        self.Bw = Bw
        self.lamda_a = lamda_a
        self.modelName = modelName
        name = 'MyProblem'  # 初始化name(函数名称，可以随意设置)
        M = 1  # 初始化M(目标维数)
        maxormins = [1]  # 初始化maxormins(目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标)
        Dim = 1  # 初始化Dim(决策变量维数)
        varTypes = [0] * Dim  # 这是一个list,初始化varTypes(决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的)
        lb = [pMin]  # 决策变量下界
        ub = [pMax]  # 决策变量上界
        lbin = [1]  # 决策变量下边界(0表示不包含该变量的下边界，1表示包含)
        ubin = [1]  # 决策变量上边界(0表示不包含该变量的上边界，1表示包含)
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # def aimFunc(self, pop):  # 目标函数
    #     Vars = pop.Phen  # 得到决策变量矩阵
    #     x1 = Vars[:, [0]]  # 取出第一列得到所有个体的x1组成的列向量
    #     x2 = Vars[:, [1]]  # 第二列
    #     x3 = Vars[:, [2]]  # 第三列
    #     pop.ObjV = 4 * x1 + 2 * x2 + x3  # 计算目标函数值，赋值给pop种群对象的ObjV属性
    #     # 采用可行性法则处理约束，numpy的hstack()把x1、x2、x3三个列向量拼成CV矩阵
    #     pop.CV = np.hstack([2 * x1 + x2 - 1, x1 + 2 * x3 - 2, np.abs(x1 + x2 + x3 - 1)])
    #     '''
    #     约束条件1，即2*x1 + x2 - 1<= 0或者2*x1 + x2 <= 1,如果是2*x1 + x2 >= 1,则取负写作(-2*x1-x2+1)
    #     '''

    # def calReferObjV(self):  # 设定目标数参考值(本问题目标函数参考值设定为理论最优值),这个函数其实可以不要
    #     referenceObjV = np.array([[2.5]])
    #     return referenceObjV

    # @ea.Problem.single
    def evalVars(self, Vars):  # 定义目标函数（含约束）
        if self.modelName == 'snc':
            f = get_stepForParam(get_dcp_snc(Vars, lamda_a=self.lamda_a, Bw=self.Bw)[0],
                                 pow(10, Vars / 10) * h, self.loops, v1=self.v1, v2=self.v2, W=self.Bw)[2]  # 计算目标函数值
        else:
            f = get_stepForParam(get_dcp_queue(Vars, lamda_a=self.lamda_a, Bw=self.Bw)[0],
                                 pow(10, Vars / 10) * h, self.loops, v1=self.v1, v2=self.v2, W=self.Bw)[2]
        # CV = np.array()  # 计算违反约束程度
        # print(f.reshape(1, 1))
        return f.reshape(1, 1)


def geat(loops: int, lamda_a: float, v1: float, v2: float, modelName: str, pMim: float, pMax: float, Bw: float
         ) -> float:
    """================================实例化问题对象==========================="""
    problem = MyProblem(loops, pMim, pMax, Bw, lamda_a, v1, v2, modelName)  # 生成问题对象
    # 构建算法
    algorithm = ea.soea_SEGA_templet(problem,
                                     ea.Population(Encoding='RI', NIND=1),     # 种群规模
                                     MAXGEN=1,  # 最大进化代数
                                     logTras=1,     # 表示每隔多少代记录一次日志信息，0表示不记录
                                     trappedValue=1e-1)
    # 求解
    res = ea.optimize(
        algorithm, seed=1, verbose=False, drawing=0, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')
    return res.get('Vars').item()
    # """==================================种群设置==============================="""
    # Encoding = 'RI'  # 编码方式
    # NIND = 100  # 种群规模
    # Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    # population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象(此时种群还没被初始化，仅仅是完成种群对象的实例化)
    # """================================算法参数设置============================="""
    # myAlgorithm = ea.soea_DE_rand_1_L_templet(problem, population)  # 实例化一个算法模板对象
    # myAlgorithm.MAXGEN = 500  # 最大进化代数
    # myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
    # myAlgorithm.recOper.XOVR = 0.7  # 重组概率
    # """===========================调用算法模板进行种群进化======================="""
    # res = ea.optimize(myAlgorithm)
    # population.save()  # 把最后一代种群的信息保存到文件中
