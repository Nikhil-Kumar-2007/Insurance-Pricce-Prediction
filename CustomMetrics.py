import numpy as np


def custom_mean_absolute_error(y_test, y_pred):
    return np.mean(abs(y_test- y_pred))



def custom_mean_squared_error(y_test, y_pred):
    return np.mean((y_test- y_pred)**2)  



def custom_mean_squared_error(y_test, y_pred):
    return np.mean(((y_test- y_pred)**2)**0.5)        



def custom_r2_score(y_test, y_pred):
    ssr = sum((y_test - y_pred)**2) 
    ssm = sum((y_test - y_test.mean())**2) 
    r2_score = 1 - (ssr/ssm)
    return r2_score



def custom_adjusted_r2_score(x_test, y_test, y_pred):
    ssr = sum((y_test - y_pred)**2) 
    ssm = sum((y_test - y_test.mean())**2) 
    r2_score = 1 - (ssr/ssm)

    n_samples, n_features = x_test.shape
    adjusted_r2_score = 1 - (((1-r2_score)*(n_samples-1))/(n_samples-n_features-1))
    return adjusted_r2_score