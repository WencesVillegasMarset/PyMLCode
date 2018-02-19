"""
Created on Mon Feb 19 16:04:18 2018

@author: Wences
"""
#Generate nonlinearly separable dataset resembling a XOR function
import numpy as np
import matplotlib.pyplot as plt
from perceptron import plot_decision_regions as pdr
#seed the generator
np.random.seed(5)
#Create random matrix of 200 samples and 2 columns
X_xor = np.random.randn(200, 2)
#get the y matrix from the X matrix, getting the truth values from each column of X
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1] > 0)
#where xor gives true replace with 1 else with -1
y_xor = np.where(y_xor, 1, -1)
#make a scatter plot for the dataset generated
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='s', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='x', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10, gamma=0.1, random_state=0)
svm.fit(X_xor, y_xor)
pdr(X_xor, y_xor, classifier=svm)
plt.legend()
plt.show()