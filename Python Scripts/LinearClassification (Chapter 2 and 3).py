# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

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

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    #Markers and colors
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot decision region
    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    #build array with 2 rows (xx1 and xx2) then transpose it to get a 2 column matrix
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1], c='', alpha=1.0, linewidths=1, marker='o', 
                    s=55, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))

y_combined = np.hstack((Y_train, Y_test))

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm,
                      test_idx=range( np.size(X, 0)-np.size(X_test, 0),np.size(X,0)))
plt.xlabel('petal lenght standardized')
plt.ylabel('petal width standardized')
plt.legend(loc='upper left')
plt.show()
    
    
    
    
    
    