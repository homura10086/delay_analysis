import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.stats import poisson, expon
from numpy import random

np.random.seed(1)
size = int(1e2)
N0 = pow(10, -160 / 10)  # mW/Hz
sigma = sqrt(1 / 2)  # 瑞利分布参数
dis = random.normal(loc=400, scale=30, size=size)  # m
# f = 2.6  # Ghz
sigma_ls = pow(10, 6 / 10)  # 阴影衰落标准差 6dB, 比值
a = 3.0  # 路径损耗因子(2~6)
Pl = np.power(dis, a)  # 比值
G = 1 / (Pl * sigma_ls)  # 比值
Bw = 1e6  # Hz
lamda_a = 1e3  # 到达率 packages/s
h = G / (N0 * Bw)  # 比值

arrival = poisson.rvs(
    mu=lamda_a,
    size=size,
    random_state=1,
)
channel = h * expon.rvs(
    scale=1 / (2 * (sigma ** 2)),
    size=size,
    random_state=1,
)

#   根据到达率和信道环境进行聚类
#   自适应选择分簇数
scores = []
x = np.hstack((arrival, channel)).reshape(2, size).transpose()

for k in range(2, 11):

    k_means = KMeans(
        n_clusters=k,
        random_state=1
    )
    k_means.fit(x)
    y_predict = k_means.predict(x)
    scores.append(metrics.calinski_harabasz_score(x, y_predict))

num = np.argmax(scores) + 2
k_means = KMeans(
        n_clusters=num,
        random_state=1
    )
k_means.fit(x)
y_predict = k_means.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_predict)
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c='r', marker='*')
plt.grid()
plt.show()
print(num)
print(k_means.inertia_)     # sse误差平方和/平均畸变程度
print(metrics.calinski_harabasz_score(x, y_predict))    # 衡量分类情况和理想分类情况之间的区别
print(metrics.silhouette_score(x, y_predict, random_state=1))   # 轮廓系数
print(k_means.cluster_centers_)
