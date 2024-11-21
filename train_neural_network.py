from TrainingSet import load_data  # 加载数据函数
from NeuralNetworkArchitecture import NeuralNetwork  # 导入神经网络类
import numpy as np

# Step 1: 加载训练和验证数据
X_train, y_train = load_data()
input_dim = X_train.shape[1]  # 特征数量
output_dim = 1  # 输出层大小（适用于二分类）

# Step 2: 定义超参数范围
learning_rates = [0.001, 0.01, 0.1]
hidden_dims = [5, 10, 20]
epochs_list = [500, 1000, 1500]

best_hyperparams = None
best_loss = float('inf')

# Step 3: 遍历所有超参数组合
for learning_rate in learning_rates:
    for hidden_dim in hidden_dims:
        for epochs in epochs_list:
            print(f"Training with: learning_rate={learning_rate}, hidden_dim={hidden_dim}, epochs={epochs}")

            # 初始化模型
            model = NeuralNetwork(input_dim, hidden_dim, output_dim)

            # 训练模型
            for epoch in range(epochs):
                output = model.forward(X_train)  # 前向传播
                loss = model.compute_loss(output, y_train)  # 计算损失
                model.backward(X_train, y_train, learning_rate)  # 反向传播并更新权重

            # 输出最终损失
            final_loss = np.mean(loss.numpy())  # 将损失转换为可读数值
            print(f"Final Loss: {final_loss:.4f}")

            # 如果当前组合的损失更低，更新最佳超参数
            if final_loss < best_loss:
                best_loss = final_loss
                best_hyperparams = (learning_rate, hidden_dim, epochs)

print("Hyperparameter tuning complete.")
print(f"Best Hyperparameters: learning_rate={best_hyperparams[0]}, hidden_dim={best_hyperparams[1]}, epochs={best_hyperparams[2]}")
