import numpy as np
'''
Script pertaining to Problem 3, parts (b) and (d)
'''


def calc_z(Z, X, Y):
    return (np.sum(X:,1] == Y) & (X[:,2] == Z)) / np.sum((X[:,1]==Y))

# generate uniform seed 
unif = np.random.uniform(low = 0, high - 1, size=(1000,3))

X = []
Y = []
Z = []

for i in range(0, len(unif)):
    if (unif[i] < 0.5):
        X.append(unif[i])
    elif(unif[i] < np.exp(4*x - 2) / (1 + np.exp(4*x - 2))):
        Y.append(unif[i])
    elif(unif[i] < np.exp(2*(x + y) - 2) / (1 + np.exp(2*(x + y) - 2))):
        Z.append(unif[i])

# convert to np array
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)


    
