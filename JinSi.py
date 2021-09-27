import matplotlib.pyplot as plt
from scipy.stats import chi2
from math import *
import numpy as np

gs = []
gs_mean = []
hs = []
d = 10
a = 2
h_ = 1
for k in range(1, 100):
    scale = 1 / (pow(d, a) * (k + 1))
    g_mean = chi2.mean(df=2, loc=k) * scale
    g = chi2.rvs(df=2, loc=k) * scale
    gs.append(g)
    gs_mean.append(g_mean)

h = np.array(pow(h_, 2) / pow(d, a)).repeat(len(gs))
plt.plot(gs, color='b')     # 随机
plt.plot(gs_mean, color='black')    # 平均
plt.plot(h, color='r')  # 大尺度
plt.legend(['Include small-scale fading', 'Average channel gain', 'Only large-scale fading'])
plt.xlabel('Rician factor')
plt.ylabel('Channel gain')
plt.grid()
plt.show()
