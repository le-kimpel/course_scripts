import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import notebook
import matplotlib.pyplot as plt
import pandas as pd

def k_fold_classifier(classifier, X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    accuracy=np.empty(n_splits)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        classifier.fit(X[train_index], Y[train_index])
        Y_pred = classifier.predict(X[test_index])
        accuracy[i] = np.sum(Y_pred==Y[test_index]) / len(Y_pred)
    return accuracy.mean()

def kernel(p):
    def f(x,y):
        N = (1 + x@y.T)
        return np.sign(N) * (np.abs(N))**p
    return f

def accuracy(p, X_scaled):
    classifier = SVC(kernel=kernel(p), max_iter=1000000, random_state=0)
    return k_fold_classifier(classifier, X_scaled, Y, n_splits=5)

if __name__=="__main__":
    df = pd.read_csv("spam.txt", header=None, delim_whitespace=True)
    # get the numpy array vers of the cols
    X, Y = df.loc[:, df.columns != 57].to_numpy(), df.loc[:,57].to_numpy()
    X_scaled= StandardScaler().fit_transform(X)
    res = {}
    step = 0.1
    for p in notebook.tqdm(np.arange(-3, 3 + step, step=step)):
        res[p] = accuracy(p, X_scaled)
    res = np.array([(a,b) for a, b in res.items()]).T

    plt.plot(res[0], res[1])
    plt.xlabel("Value of p")
    plt.ylabel("Accuracy")
    plt.show()

    max_ind = np.argmax(res[1])
    max_p = res[0, max_ind]
    acc = res[1, max_ind]

    print("P: " + str(max_p))
    print("Cross-validation accuracy: " + str(acc))
