# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:19:30 2018

@author: Wences
"""
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#Setup marker list and color list to feed into colormap
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    #create color map converting the first 2 colors to RGB useful values (we only have 2 classes)
    colormap = ListedColormap(colors[:len(np.unique(y))])
    # get min and max from each feature, this lets us define the surface of the plot where we want to visualize the dec. boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # generate meshgrid to map function to multiple coordinates within the area of interest
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # feed all those pairs of x,y coords, to the classifier raveling the xx1 and xx2 matrixes(cant feed whole matrices)
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # shape the whole array of predictions to the shape of the matrix xx1
    Z = Z.reshape(xx1.shape)
    # pass the matrices and the Z values to plot a filled contour plot of the decision boundary
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=colormap)
    # establish the limits of the contour visualization to the same range of the points tested
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples on the same plot

    for idx, cl in enumerate(np.unique(y)):#build scatter plots for each class
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colormap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)
    #highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx[0]:test_idx[1]]
        plt.scatter(X_test[:,0], X_test[:,1], c='', alpha=1.0, linewidths=1, marker='o', 
                    s=55, label='test set')