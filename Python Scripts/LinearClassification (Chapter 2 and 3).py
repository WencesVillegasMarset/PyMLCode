# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from plottingFunctions import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, Y_train)
print(svm.score(X_test_std, Y_test)*100)


#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(C=1000, random_state=0)
#lr.fit(X_train_std, Y_train)
#print(lr.score(X_test_std, Y_test)*100)
#from sklearn.metrics import accuracy_score


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((Y_train, Y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm,
                      test_idx=range( np.size(X, 0)-np.size(X_test, 0),np.size(X,0)))
plt.xlabel('petal lenght standardized')
plt.ylabel('petal width standardized')
plt.legend(loc='upper left')
plt.show()
    
    
    
    
    
    