# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 22:54:34 2018

@author: Wences
"""
from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets as datasets
import numpy as np
from sklearn.cross_validation import train_test_split 
from plottingFunctions import plot_decision_regions 
import matplotlib.pyplot as plt
#get that data
dset = datasets.load_iris()
X = dset.data[:,[2,3]]
y = dset.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
#Create and train that tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(np.size(X_train,0),np.size(X,0)))
plt.xlabel('petal lenght cm')
plt.ylabel('petal width cm')
plt.legend(loc='upper left')
plt.show()
