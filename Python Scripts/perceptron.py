import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from plottingFunctions import plot_decision_regions

class Perceptron():
    '''
    Perceptron classifier implementation

    Initialization parameters:
    l_rate : float
        Learning rate
    n_iters : int
        Number of epochs to run
    r_seed : float
        Random seed to initialize weights
    Attributes
    w_vector : 1d-array
        Vector of weights after training
    error_list : list
        Number of errors in classification per epoch

    '''
    def __init__(self, l_rate, n_iters, r_seed=1):
        self.l_rate = l_rate
        self.n_iters = n_iters
        self.r_seed = r_seed
    def train(self, X, y):
        '''
        X is a matrix of M x N where
        Y is an array of M x 1
            M is number of samples
            N is number of features                                 
        Trains classifier initializing weights and creating list of errors per epoch ran
        Returns Perceptron with trained weights and error per epoch info
        '''
        #initialize weights plus  bias
        randomGen = np.random.RandomState(seed=self.r_seed)

        self.w_vector = randomGen.normal(loc=0.0, scale=0.01, size=1 +  np.shape(X)[1])
        #initialize error list
        self.error_list = []
        #optimization loop: epochs over training set
        for _ in range(self.n_iters):
            errors = 0
            for sample, target in zip(X, y): #Loop over training set, selecting a row and respective target class for each case
                update = self.l_rate * (target - self.predict(sample))
                self.w_vector[0] += update
                self.w_vector[1:] += (update * sample)
                #if an update is != 0 then register error
                if update != 0.0: 
                    errors += 1

            self.error_list.append(errors)
        return self
        
    def net_input(self, X):
        '''
        Calculate net input Z defined as X*W to feed into squashing function
        '''
        return np.dot(X, self.w_vector[1:]) + self.w_vector[0] 
    def predict(self, X):
        '''Return class label prediction from input given'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)
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
perc = Perceptron(l_rate=0.1, n_iters=10)
perc.train(X, y)
print(str(perc.predict([4,4])) + " Predicted for 1 and 4")
print(perc.error_list)

#plot error list
plt.plot(range(1,len(perc.error_list) + 1), perc.error_list, marker='.')
plt.show()



plot_decision_regions(X, y, classifier=perc)
plt.xlabel('sepal length cm')
plt.ylabel('petal length cm')
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
