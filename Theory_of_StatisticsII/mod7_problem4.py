import numpy as np
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt

data = np.array([[[35, 59], [47, 112]], [[42, 77], [26, 76]]])

# part (a): get the MLE's for the multinomial data: (malignant, died, Boston), (malignant, survived, Boston), (benign, died, Boston), (benign, survived, Boston)...and the same with Glamorgan.
# the estimated probabilities for each of these. 
total = 0
for i in data:
   for j in i:
      total += sum(j)
print(total)
mle = data/total
print("MLE: " + str(mle))

# part (b): compute counts of X divided by the total in each category
# print out the standard error
variance = ((0.255*0.755)/102)
se = np.sqrt(variance)
print("Standard error calculation for P(X3 = dead | X2 = benign, X1 = Glamorgan: " + str(se))

print(data)

# part(c): now perform chi2 tests

# get the contingency tables relevant to each test
for i in range(data.shape[2]):
   T1 = data[:,:,i]
   chi_1, p_1, dof1, expected1 = chi2_contingency(T1)
  
for i in range(data.shape[1]):
   T2 = data[:,i,:]
   chi_2, p_2, dof2, expected2 = chi2_contingency(T2)

for i in range(data.shape[1]):
   T3 = data[i,:,:]
   chi_3, p_3, dof3, expected3 = chi2_contingency(T3)

print("Pval 1 (X1 ind. X2 | X3): " + str(p_1))
print("Pval 2: (X1 ind. X3 | X2): " + str(p_2))
print("Pval 3: (X2 ind. X3 | X1): " + str(p_3))

G = nx.Graph()
G.add_node("X1")
G.add_node("X3")
G.add_node("X2")
G.add_edge("X1", "X3")
G.add_edge("X3", "X2")

labeldict = {}
labeldict["X1"] = "X1"
labeldict["X2"] = "X2"
labeldict["X3"] = "X3"

nx.draw(G, labels = labeldict, with_labels=True)
plt.show()
