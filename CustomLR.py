import numpy as np
from itertools import combinations_with_replacement



class LinearRegressionOLS:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.beeta_ = None

    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis = 1)
        self.beeta_ = np.linalg.pinv(np.dot(x_train.T, x_train)) @ np.dot(x_train.T, y_train)
        self.intercept_ = self.beeta_[0]
        self.coef_ = self.beeta_[1:]

    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        return np.dot(x_test, self.beeta_)




class PolynomialLinearRegression:
    def __init__(self, degree = 2):
        self.degree = degree
        self.intercept_ = None
        self.coef_ = None
        self.beta_ = None

    def create_polynomial_faetures(self, x):
        n_samples, n_features = x.shape
        self.combinations_list = []
        for deg in range(self.degree+1):
            self.combinations_list.extend(list(combinations_with_replacement(range(n_features), deg)))
        self.x_poly = np.ones((n_samples, len(self.combinations_list)))    
        
        for i, comb in enumerate(self.combinations_list):
            if (len(comb) > 0):
                self.x_poly[:, i] = np.prod(x[:, comb], axis = 1)
        return self.x_poly
        
    def fit(self, x_train, y_train):    
        self.x_train_poly = self.create_polynomial_faetures(x_train)
        self.beta_ = np.dot(np.linalg.pinv(self.x_train_poly), y_train) 
        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]

    def predict(self, x_test):
        self.x_test_poly = self.create_polynomial_faetures(x_test)
        return np.dot(self.x_test_poly, self.beta_)




class LinearRegressionBatchGD:
    
    def __init__(self, learning_rate = 0.001, epochs = 1000):
        
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        
    def fit(self,x_train,y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(x_train.shape[1])
        
        for i in range(self.epochs):
            y_hat = np.dot(x_train,self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            self.intercept_ = self.intercept_ - (self.lr * intercept_der)
            
            coef_der = (-2/x_train.shape[0]) * np.dot((y_train - y_hat),x_train)
            self.coef_ = self.coef_ - (self.lr * coef_der)
        
    
    def predict(self,x_test):
        return np.dot(x_test,self.coef_) + self.intercept_




class SGDRidgeRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, alpha = 0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.intercept_ = 0
        self.coef_ = np.ones(n_features)
        for i in range(self.epochs):
            for j in range(n_samples):
                self.idx = np.random.randint(low = 0, high = n_samples)
                self.subtr = y_train - (self.intercept_ + np.dot(x_train, self.coef_))
                
                self.intercept_grad = -2 * np.mean(self.subtr)
                self.intercept_ = self.intercept_ - (self.lr * self.intercept_grad)

                self.coef_grad = (-2/n_samples) * (np.dot(self.subtr, x_train)) + self.alpha * self.coef_
                self.coef_ = self.coef_ - (self.lr * self.coef_grad)

    def predict(self, x_test):
        return np.dot(x_test,self.coef_) + self.intercept_        



class SGDLassoRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, alpha = 0.01):
        self.lr = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.intercept_ = 0
        self.coef_ = np.ones(n_features)
        for i in range(self.epochs):
            for j in range(n_samples):
                self.idx = np.random.randint(low = 0, high = n_samples)
                self.subtr = y_train - (self.intercept_ + np.dot(x_train, self.coef_))
                
                self.intercept_grad = -2 * np.mean(self.subtr)
                self.intercept_ = self.intercept_ - (self.lr * self.intercept_grad)

                self.coef_grad = (-2/n_samples) * (np.dot(self.subtr, x_train)) + self.alpha * np.sign(self.coef_)
                self.coef_ = self.coef_ - (self.lr * self.coef_grad)

    def predict(self, x_test):
        return np.dot(x_test,self.coef_) + self.intercept_ 



class SGDElasticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000, alpha = 1.0, l1_ratio = 0.5):
        self.lr = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.intercept_ = 0
        self.coef_ = np.ones(n_features)
        for i in range(self.epochs):
            for j in range(n_samples):
                self.idx = np.random.randint(low = 0, high = n_samples)
                self.subtr = y_train - (self.intercept_ + np.dot(x_train, self.coef_))

                l1_penality = self.l1_ratio * np.sign(self.coef_)
                l2_penality = (1 - self.l1_ratio) * self.coef_
                self.intercept_grad = -2 * np.mean(self.subtr)
                self.intercept_ = self.intercept_ - (self.lr * self.intercept_grad)

                self.coef_grad = (-2/n_samples) * (np.dot(self.subtr, x_train)) + self.alpha*(l1_penality+l2_penality)
                self.coef_ = self.coef_ - (self.lr * self.coef_grad)

    def predict(self, x_test):
        return np.dot(x_test,self.coef_) + self.intercept_                 