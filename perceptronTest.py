import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
import numpy as np
import perceptron



#loading iris dataset into a dataframe 
df  = pd.read_csv('https://archive.ics.uci.edu/ml/'
        'machine-learning-databases/iris/iris.data', header=None)
#select only 2 classes
#get values of the column containing classes
y = df.iloc[0:100, 4].values 
#print(y)
#replace classes for useful numerical values
y = np.where(y == 'Iris-setosa', -1 , 1) 
#print(y)
#Extract only 2 features (sepal lenght and petal lenght)
X = df.iloc[0:100, [0,2]].values
#Scatter plot of the data selected, 
plt.scatter(X[:49, 0], X[:49, 1], label='setosa', color='green')
plt.scatter(X[50:, 0], X[50:, 1], label='versicolor', color='red', marker='x')
#label for first feature (X1)
plt.xlabel('sepal length in cm ')
#label for second feature (X2)
plt.ylabel('petal length in cm')
plt.legend()
#show the plot
plt.show()

#Data is ready, now create perceptron classifier 
perc = perceptron.Perceptron(l_rate=0.1, n_iters=10)
perc.train(X, y)
print(str(perc.predict([4,4])) + " Predicted for 1 and 4")
print(perc.error_list)

#plot error list
plt.plot(range(1,len(perc.error_list) + 1), perc.error_list, marker='.')
plt.show()

#define function to plot preety decision boundary
def plot_decision_regions(X, y , classifier, resolution=0.02):
    #Setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')

    colormap = ListedColormap(colors[:len(np.unique(y))])
    #get min and max from each feature
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #generate meshgrid to map function
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=colormap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #plot class samples

    for idx, cl in enumerate(np.unique(y)):#build scatter plots for each class
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colormap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

plot_decision_regions(X, y, classifier=perc)
plt.xlabel('sepal length cm')
plt.ylabel('petal length cm')
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
