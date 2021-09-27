import numpy as np
import matplotlib
from scipy.stats import bernoulli

gamma = bernoulli.rvs(p=0.5, size=10)
print(type(gamma))
