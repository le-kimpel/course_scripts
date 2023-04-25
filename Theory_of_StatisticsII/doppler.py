from scipy import stats, integrate
import pywt
import numpy as np
import math
import matplotlib.pyplot as plt

def doppler(x, sigma):
    return math.sqrt(x * (1-x)) * math.sin(float(2.1 * math.pi)/float(x + 0.05)) + sigma * np.random.normal(0,1,1)[0]

def integrand(x, sigma, j):
    return math.sqrt(x * (1-x)) * math.sin(float(2.1 * math.pi)/float(x + 0.05)) * math.sqrt(2) * math.cos(math.pi * j * x)

def cosine_basis(J, sigma):
    I = []
    sum_ = 0
    for j in range(1, J+1):
        sum_ += integrate.quad(integrand, 0, 1, args=(sigma, j))[0]
        I.append(sum_)
    return I

# initial setup
n = 1024
sigma = 0.1
X = []
for i in range(1, n+1):
    X.append(float(1)/float(i))

# generate data
Y = [doppler(x, sigma) for x in X]

# use values of J from 10...100
J = np.arange(10, 110, 10)
for j in J:

    # fit the curve using the cosine basis method
    curve = np.array(cosine_basis(j, sigma))
    M = np.arange(1, j+1, 1)

    fig, ax = plt.subplots()
    ax.plot(M, curve)
    # build the confidence bands...
    ax.fill_between(M, curve - 0.05*curve , curve + curve*(0.05), color='r', alpha=0.2)
    plt.show()

# now use Haar wavelets
cA, cD = pywt.dwt(Y, 'Haar')
plt.plot(cA)
plt.xlabel("Num. Samples")
plt.ylabel("Approximation Coefficient (cA)")
plt.show()


plt.plot(cD)
plt.xlabel("Num. Samples")
plt.ylabel("Detailed Coefficient (cD)")
plt.show()
