import numpy as np
'''
Script pertaining to Problem 3, parts (b) and (d)
'''

def calc_z(Z, X, Y):
    return np.sum((X[:, 1] == Y) & (X[:, 2] == Z)) / np.sum((X[:, 1] == Y))

# generate uniform seed 
unif = np.random.uniform(low = 0, high - 1, size=(1000,3))

# now generate a bunch of random samples
x = np.where(seeds[:, 0] < 1/2, 1, 0)
y = np.where(seeds[:, 1] < np.exp(4*x - 2) / (1 + np.exp(4*x - 2)), 1, 0)
z = np.where(seeds[:, 2] < np.exp(2*(x + y) - 2) / (1 + np.exp(2*(x + y) - 2)), 1, 0)

X = np.zeros((n, 3), dtype=int)
X[:, 0] = x
X[:, 1] = y
X[:, 2] = z

estimates = []

# estimate P(Z = 1 | Y = 1) 
for i in range(1,1000):
    z_est = calc_z(X[i], 1, 1)
    estimates.append(z_est)
    
