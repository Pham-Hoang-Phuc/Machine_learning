import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def computing_cost(self, x, y, w, b):
        
        m = x.shape[0]
        cost = 0
        
        for i in range(m):
            f_wb = w * x[i] + b
        
            cost += (f_wb - y[i]) ** 2
        cost = cost / (2 * m)
        return cost
    
    def inplement_gradien(self, x, y, w, b):
        
        m = x.shape[0]
        df_dw = 0
        df_db = 0
        
        for i in range(m):
            f_wb = w * x[i] + b
            df_dw_i = (f_wb - y[i]) * x[i]
            df_db_i = (f_wb - y[i])
            
            df_dw += df_dw_i
            df_db += df_db_i
        
        df_dw = df_dw / m
        df_db = df_db / m
        return df_dw, df_db

    def fit(self, x, y, w_in=0, b_in=0, learning_rate=0.01, iteration=1000):

        w = w_in
        b = b_in
        J_hist = []
        
        print("______Trainning______")
        for i in range(iteration):
            
            df_dw, df_db = self.inplement_gradien(x, y, w, b)
            
            w = w - learning_rate * df_dw
            b = b - learning_rate * df_db
            
            cost = self.computing_cost(x, y, w, b)
            J_hist.append(cost)
            if i % (iteration / 10) == 0:
                print(f"Iteration {i}, cost: {float(J_hist[i])}")
                
        self.w = w
        self.b = b
        return w, b, J_hist
    
    def predict(self, x):
        
        if self.w is None or self.b is None:
            print("Model is not learning")
            return
        
        y_pred = x * self.w + self.b
        return y_pred[0]
        
    
if __name__ == "__main__":
    lr = LinearRegression()
    
    california_housing = fetch_california_housing(as_frame=True)
    X = np.array([[1.0], [2.5], [3.0], [4.5], [5.0], [6.5], [7.0], [8.5], [9.0], [10.0]])
    Y = np.array([3.1, 5.2, 6.8, 8.9, 10.3, 13.0, 14.5, 16.7, 18.2, 19.9])
    
    w, b, J_hist = lr.fit(X, Y, iteration=100, learning_rate=0.001) 
    
    iteration_arr = np.arange(100)
    
    plt.plot(iteration_arr, J_hist)
    plt.show()
    
    y_pred = lr.predict(3.5)
    print()
    print("Predict: ", y_pred)