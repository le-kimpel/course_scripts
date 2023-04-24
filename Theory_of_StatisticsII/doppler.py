from scipy import stats
import numpy as np
import math

def doppler(x, sigma):
    return math.sqrt(x * (1-x)) * math.sin(float(2.1 * math.pi)/float(x + 0.05)) + sigma * np.random.normal(0,1,1)


# initial setup
n = 1024
sigma = 0.1
X = []
for i in range(1, n+1):
    X.append(float(1)/float(n))
X = np.array(X)

# generate data
Y = [doppler(x, sigma) for x in X]

