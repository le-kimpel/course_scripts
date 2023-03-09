import numpy as np
import matplotlib.pyplot as plt 

'''
Script pertaining to Problem 3, parts (b) and (d)
'''

def calc_z(Z, X, Y):
    if np.sum((X[:, 1] == Y)) == 0:
        return 0.5
    return np.sum((X[:, 1] == Y) & (X[:, 2] == Z)) / np.sum((X[:, 1] == Y))

'''
PART (B) 
'''

# generate uniform seed 
unif = np.random.uniform(low = 0, high = 1, size=(1000,3))

# now generate a bunch of random samples
x = np.where(unif[:, 0] < 1/2, 1, 0)
y = np.where(unif[:, 1] < np.exp(4*x - 2) / (1 + np.exp(4*x - 2)), 1, 0)
z = np.where(unif[:, 2] < np.exp(2*(x + y) - 2) / (1 + np.exp(2*(x + y) - 2)), 1, 0)

X = np.zeros((1000, 3), dtype=int)
X[:, 0] = x
X[:, 1] = y
X[:, 2] = z

estimates = []

# estimate P(Z | Y = 1) 
for i in range(0,1000):
    z_est = calc_z(1,X[:i], 1)
    estimates.append(z_est)


plt.scatter(np.arange(0,1000, 1, dtype=int), estimates)
plt.ylabel('P(Z | Y = 1)')
plt.xlabel('N')
plt.show()


'''
PART (D) 
'''

# generate uniform seed 
unif2 = np.random.uniform(low = 0, high = 1, size=(1000,3))
y_intervention = 1

# now generate a bunch of random samples
x = np.where(unif2[:, 0] < 1/2, 1, 0)
y = y_intervention * np.ones(1000,dtype=int)
z = np.where(unif2[:, 2] < np.exp(2*(x + y) - 2) / (1 + np.exp(2*(x + y) - 2)), 1, 0)

X_prime = np.zeros((1000, 3), dtype=int)
X_prime[:, 0] = x
X_prime[:, 1] = y
X_prime[:, 2] = z

estimates2 = []

# estimate P(Z | Y = 1) 
for i in range(0,1000):
    z_est = calc_z(1,X_prime[:i], 1)
    estimates2.append(z_est)


plt.scatter(np.arange(0,1000, 1, dtype=int), estimates2)
plt.ylabel('P(Z | Y = 1)')
plt.xlabel('N')
plt.show()
