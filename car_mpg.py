import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model
from itertools import chain, combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

'''
Compute the powerset of some set (found in itertools documentation)
'''
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


'''
 zheng-loh model selection
'''
def zheng_loh(X,Y,var, covars):
    
    X = X.copy()
    y = Y.copy()
    # first, compute the full model
    fullmodel = LinearRegression()
    fullmodel.fit(X,y)

    # Least squares solution
    beta_hat = (np.linalg.inv(X.T @ X) @ X.T @ Y).to_numpy()

    # Predicted solutions
    Y_pred = X @ beta_hat

    # Prediction errors
    epsilon_hat = Y_pred - Y

    # Error on training data
    rss = epsilon_hat.T @ epsilon_hat

    # array for coefs
    N = len(X)
    p = len(X.columns) + 1
    std_error = []
    zl = []
    
    X_with_intercept = np.empty(shape=(N, p), dtype=np.cfloat)
    X_with_intercept[:, 0] = 1
    X_with_intercept[:, 1:p] = X.values

    
    y_hat = fullmodel.predict(X)
    residuals = y.values - y_hat
    residual_sum_of_squares = residuals.T @ residuals
    sigma_squared_hat = residual_sum_of_squares/ (N - p)
    var_beta_hat = np.linalg.inv(X.T @ X) * sigma_squared_hat
    
    Wald_stats = []

    # get vals for standard error
    for p_ in range(0,4):
        std_error.append(var_beta_hat[p_, p_] ** 0.5)

    # now calc the wald stats
    for i in range(0,4):
        Wald_stats.append(abs((beta_hat[i] / std_error[i])))
        
    # sort the results in descending order
    s = sorted(Wald_stats, reverse=True)

    for j in range(0,len(s)):
        # compute the model with the j largest Wald statistics
        for S in powerset(covars):
            if len(S) > 0:
                Z = X[list(S)].copy()
                # Create new column with all 1s for intercept at start
                Z.insert(0, 'const', 1)
            else:
                Z = pd.DataFrame({'const': np.ones_like(Y)})
            
            temp_model = LinearRegression()
            temp_model.fit(Z,y)
            zl.append((S,get_zheng_eqn(Z,Y,S,var,j)))
    # now get the minimum of this        
    return zl


'''
zheng loh model selection equation computation
avoid having to search through all possible models

Here, S is the model with the j largest Wald statistics
'''
def get_zheng_eqn(X,Y, S, var,j):
    X = X.copy()
    Y = Y.copy()
    
    # Least squares solution
    beta_hat = (np.linalg.inv(X.T @ X) @ X.T @ Y).to_numpy()

    # Predicted solutions
    Y_pred = X @ beta_hat

    # Prediction errors
    epsilon_hat = Y_pred - Y

    n = Y.shape[0]

    # Error on training data
    rss = epsilon_hat.T @ epsilon_hat

    return rss + j * var * np.log(n)

'''
Calculate BIC

'''
def bic(X,Y,S, var):
    if len(S) > 0:
        X = X[list(S)].copy()
        # Create new column with all 1s for intercept at start
        X.insert(0, 'const', 1)
    else:
        X = pd.DataFrame({'const': np.ones_like(Y)})
    
    # Least squares solution
    beta_hat = (np.linalg.inv(X.T @ X) @ X.T @ Y).to_numpy()

    # Predicted solutions
    Y_pred = X @ beta_hat

    # Prediction errors
    epsilon_hat = Y_pred - Y

    # Error on training data
    rss = epsilon_hat.T @ epsilon_hat
    
    n = Y.shape[0]
    k = X.shape[1]
    
    return n * np.log(rss / n) + k * np.log(n)

'''
Calculate mallow cp
'''
def mallow_cp(X,Y,S,var):
    if len(S) > 0:
        X = X[list(S)].copy()
        # Create new column with all 1s for intercept at start
        X.insert(0, 'const', 1)
    else:
        X = pd.DataFrame({'const': np.ones_like(Y)})
    
    # Least squares solution
    beta_hat = (np.linalg.inv(X.T @ X) @ X.T @ Y).to_numpy()

    # Predicted solutions
    Y_pred = X @ beta_hat

    # Prediction errors
    epsilon_hat = Y_pred - Y

    # Error on training data
    partial_training_error = epsilon_hat.T @ epsilon_hat
    
    # Increase size of S by to account for constant covariate
    return partial_training_error + 2 * (len(S) + 1) * var

    

'''
Get variance of the overall model
'''
def variance(X, Y):
    X = X.copy()
    
    # Create new column with all 1s for intercept at start
    X.insert(0, 'const', 1)
    
    # Least squares solution
    beta_hat = (np.linalg.inv(X.T @ X) @ X.T @ Y).to_numpy()

    # Predicted solutions
    Y_pred = X @ beta_hat

    # Prediction errors
    epsilon_hat = Y_pred - Y

    # Error on training data
    training_error = epsilon_hat.T @ epsilon_hat
    
    # Estimated error variance
    return (training_error / (Y.shape[0] - X.shape[1]))


# read in and parse data
covars = ['VOL', 'HP', 'SP', 'WT']
data = pd.read_csv('cars.txt', sep='\t')

print(data)

Y = data['MPG']
X = data[covars]

# obtain the linear regression for X and Y
x = sm.add_constant(X)

reg = sm.OLS(Y, X).fit()

print(reg.summary())
# some lazy graphing

'''
fig, ax = plt.subplots(figsize=(10, 5))
fig = sm.graphics.plot_fit(reg, 1, ax=ax)
ax.set_ylabel("MPG")
ax.set_xlabel("VOL")
ax.set_title("Multiple Linear Regression Results: VOL")
plt.show()


fig, ax = plt.subplots(figsize=(10, 5))
fig = sm.graphics.plot_fit(reg, 3, ax=ax)
ax.set_ylabel("MPG")
ax.set_xlabel("SP")
ax.set_title("Multiple Linear Regression Results: SP")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
fig = sm.graphics.plot_fit(reg, 2, ax=ax)
ax.set_ylabel("MPG")
ax.set_xlabel("HP")
ax.set_title("Multiple Linear Regression Results: HP")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
fig = sm.graphics.plot_fit(reg, 0, ax=ax)
ax.set_ylabel("MPG")
ax.set_xlabel("WT")
ax.set_title("Multiple Linear Regression Results: WT")
plt.show()

'''

var = variance(X,Y)

# now get the best model based on zheng loh model selection criteria
zl = zheng_loh(X,Y,var,covars)
results_zl = pd.DataFrame(zl, columns=['S', 'score'])

print("------------------------------------")
print("ZHENG-LOH")
print("------------------------------------")
print(results_zl)


curr_subset = []
curr_score = mallow_cp(X,Y,curr_subset,var)

# Next: calculation of Mallow Cp for more accurate model fitting.
# simply obtain the model's variance and then fill-in the blanks.
# now perform a forward-stepwise search, starting with an empty list
while(len(curr_subset) < len(covars)):
    best_score, best_subset = curr_score, curr_subset
    updated_set = 0
    for covar in covars:
        if (covar not in curr_subset):
            s = curr_subset + [covar]
            s_score = mallow_cp(X,Y,s, var)
            if (s_score  == min(s_score, best_score)):
                best_score, best_subset = s_score, s
                updated_set = 1
    # just give up and move on if we haven't met any criteria
    if (updated_set == 0):
        break
    curr_score, curr_subset = best_score, best_subset


print(curr_score)
print(curr_subset)

curr_subset = covars
curr_score = mallow_cp(X,Y,curr_subset,var)

# now perform a backward-stepwise search; e.g., start with the biggest model and work out way down
while(len(curr_subset) > 0):
    best_score, best_subset = curr_score, curr_subset
    updated_set = 0
    for covar in covars:
        if (covar in curr_subset):
            s = [i for i in curr_subset if i != covar]
            s_score = mallow_cp(X,Y,s, var)
            if (s_score  == min(s_score, best_score)):
                best_score, best_subset = s_score, s
                updated_set = 1
    # just give up and move on if we haven't met any criteria
    if (updated_set == 0):
        break
    curr_score, curr_subset = best_score, best_subset

print(curr_score)
print(curr_subset)
    

# now calculate using BIC.

# forward selection
curr_subset = []
curr_score = bic(X,Y,curr_subset, var)

# now perform a forward-stepwise search, starting with an empty list
while(len(curr_subset) < len(covars)):
    best_score, best_subset = curr_score, curr_subset
    updated_set = 0
    for covar in covars:
        if (covar not in curr_subset):
            s = curr_subset + [covar]
            s_score = bic(X,Y,s, var)
            if (s_score  == min(s_score, best_score)):
                best_score, best_subset = s_score, s
                updated_set = 1
    # just give up and move on if we haven't met any criteria
    if (updated_set == 0):
        break
    curr_score, curr_subset = best_score, best_subset

print(curr_score)
print(curr_subset)

curr_subset = covars
curr_score = bic(X,Y,curr_subset, var)

# now perform a backward-stepwise search; e.g., start with the biggest model and work out way down
while(len(curr_subset) > 0):
    best_score, best_subset = curr_score, curr_subset
    updated_set = 0
    for covar in covars:
        if (covar in curr_subset):
            s = [i for i in curr_subset if i != covar]
            s_score = bic(X,Y,s, var)
            if (s_score  == min(s_score, best_score)):
                best_score, best_subset = s_score, s
                updated_set = 1
    # just give up and move on if we haven't met any criteria
    if (updated_set == 0):
        break
    curr_score, curr_subset = best_score, best_subset

print(curr_score)
print(curr_subset)
    

'''
All possible regressions: mallow cp
'''

# Iterate through the powerset and calculate the score for each value
results_mallow = [(S, mallow_cp(X,Y,S,var)) for S in powerset(covars)]
    
results_mallow = pd.DataFrame(results_mallow, columns=['S', 'score'])

print("------------------------------------")
print("MALLOW")
print("------------------------------------")
print(results_mallow)

'''
All possible regressions: BIC
'''

# Iterate through the powerset and calculate the score for each value
results_bic = [(S, bic(X,Y,S,var)) for S in powerset(covars)]
    
results_bic = pd.DataFrame(results_bic, columns=['S', 'score'])

print("------------------------------------")
print("BIC")
print("------------------------------------")
print(results_bic)
