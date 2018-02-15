import numpy as np

class Perceptron():
    '''
    Perceptron classifier implementation

    Initialization parameters:
    l_rate : float
        Learning rate
    n_iters : int
        Number of epochs to run
    Attributes
    w_vector : 1d-array
        Vector of weights after training
    error_list : list
        Number of errors in classification per epoch

    '''
    def __init__(self, l_rate, n_iters):
        self.l_rate = l_rate
        self.n_iters = n_iters
    
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
        self.w_vector = np.zeros(1 +  np.shape(X)[1])
        #initialize error list
        self.error_list = []
        #optimization loop: epochs -> training set
        for _ in range(self.n_iters):
            errors = 0
            for sample, target in zip(X,y): #Loop over training set, selecting a row and respective target class for each case
                update = self.l_rate * (target - self.predict(X))
                self.w_vector[0] += update
                self.w_vector[1:] += (update * sample)
                #if an update is != 0 then register error
                if update != 0.0: 
                    errors += 1

            self.error_list.append(errors)

        
    def net_input(self, X):
        '''
        Calculate net input Z defined as X*W to feed into squashing function
        '''
        return np.dot(X, self.w_vector[1:]) + self.w_vector[0] 
    def predict(self, X):
        '''Return class label prediction from input given'''
        if self.w_vector == None:
            print('Please run the train method first to be able to predict')
        else:
            return np.where(self.net_input(X) >= 0.0, 1, -1)
