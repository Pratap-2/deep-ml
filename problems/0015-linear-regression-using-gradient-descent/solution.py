import numpy as np
def compute_coefficients(X,y_actual,y_pred):
    m = X.shape[0]
    dw = (X.T @ (y_pred-y_actual))   / m
    return dw 


def linear_regression_gradient_descent(X: np.ndarray, y_actual: np.ndarray, alpha: float, itr: int) -> np.ndarray:
    
    m, n = X.shape
    y_actual = y_actual.reshape(-1, 1)  
    theta = np.zeros((n, 1)) 
    

    for _ in range(itr):
        y_pred = X @ theta 
        error  =  ((y_pred-y_actual)**2) / 2*m
        theta -= alpha * compute_coefficients(X,y_actual,y_pred)

    

    return theta.flatten()