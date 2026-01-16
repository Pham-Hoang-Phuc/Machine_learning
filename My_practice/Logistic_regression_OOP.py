import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def build_fwb(self, X, W, b):
        z = np.dot(X, W) + b
        f_wb = self.sigmoid(z)
        return f_wb

    def computing_cost(self, X, Y, W, b):
        m = X.shape[0]
        
        f_wb = self.build_fwb(X, W, b)
        loss = Y * np.log(f_wb) + (1 - Y) * np.log(1 - f_wb)
        cost = - 1 / m * np.sum(loss)
        return cost
    
    def implememt_gradient(self, X, Y, W, b):
        m = X.shape[0]
        
        f_wb = self.build_fwb(X, W, b)
        error = f_wb - Y
        df_dw = 1 / m * np.dot(X.T, error)
        df_db = 1 / m * np.sum(error)
        
        return df_dw, df_db

    def fit(self, X, Y, alpha=0.001, iteration=1000):
        
        J_hist = []
        m, n = X.shape
        W = np.zeros((n, 1))
        b = 0
        for i in range(iteration):
            df_dw, df_db = self.implememt_gradient(X, Y, W, b)
            
            W = W - alpha * df_dw
            b = b - alpha * df_db
            
            cost = self.computing_cost(X, Y, W, b)
            J_hist.append(cost)
            
            if i % (iteration // 10) == 0:
                print(f"Iteration {i} | Cost = {cost}")
        self.w = W
        self.b = b
        return W, b, J_hist

    def predict(self, X_new):
        f_wb = self.build_fwb(X_new, self.w, self.b)

        y_pred = np.where(f_wb > 0.5, 1, 0)
        return y_pred.reshape(-1)
        
        
if __name__ == "__main__":
    # x: thời gian học tính bằng giờ
    X_data = np.array([
        [0.5], [1.0], [1.5], [2.0], [2.5], 
        [3.0], [3.5], [4.0], [4.5], [5.0]
    ])

    # Y: Kết quả thi (0 = Rớt, 1 = Đậu)
    Y_labels = np.array([
        0, 0, 0, 0, 1, 
        1, 1, 1, 1, 1
    ]).reshape(-1, 1) # Đảm bảo Y có kích thước (M, 1)

    # --- Khởi tạo và Huấn luyện Mô hình ---    
    # 1. Khởi tạo Mô hình
    model = LogisticRegression()
    
    # 2. Định nghĩa Siêu tham số
    # Tốc độ học (alpha) được tăng lên để hội tụ nhanh hơn
    LEARNING_RATE = 0.1 
    N_ITERATIONS = 5000 
    
    print("--- Bắt đầu Huấn luyện Logistic Regression ---")
    print(f"X shape: {X_data.shape}, Y shape: {Y_labels.shape}")
    print(f"Learning Rate: {LEARNING_RATE}, Iterations: {N_ITERATIONS}\n")

    # 3. Gọi hàm fit để huấn luyện mô hình
    final_w, final_b, cost_history = model.fit(
        X=X_data, 
        Y=Y_labels, 
        alpha=LEARNING_RATE, 
        iteration=N_ITERATIONS
    )
    print("Vẽ biểu đồ")
    iteration_range = np.arange(N_ITERATIONS)
    plt.plot(iteration_range, cost_history)
    plt.show()

    # 4. Hiển thị kết quả cuối cùng
    print("\n--- Kết Quả Sau Huấn Luyện ---")
    print(f"Trọng số W cuối cùng: {final_w}")
    print(f"Độ lệch b cuối cùng: {final_b}")
    print(f"Cost cuối cùng: {cost_history[-1]}")
    
   # --- Kiểm tra Dự đoán (Ví dụ) ---
    test_x = np.array([[1.2], [4.8]])
    print("\n--- Dự đoán cho mẫu kiểm tra ---")
    # Tính xác suất (sử dụng hàm sigmoid của model)
    y_pred = model.predict(test_x)
    print(f"Với dữ liệu: \n{test_x.reshape(-1,1)}")
    print("Du doan: ", y_pred)        