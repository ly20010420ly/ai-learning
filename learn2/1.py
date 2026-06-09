import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("手写线性回归 - 梯度下降法")
print("=" * 60)

# 1. 生成模拟数据
np.random.seed(42)  # 固定随机种子，结果可重复
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5

print(f"数据形状: X={X.shape}, y={y.shape}")

# 2. 数据可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.7)
plt.xlabel("X")
plt.ylabel("y")
plt.title("原始数据分布")
plt.grid(True, alpha=0.3)

# 3. 添加偏置项（x0=1）
X_b = np.c_[np.ones((100, 1)), X]  # 添加一列1
print(f"添加偏置后形状: {X_b.shape}")


# 4. 梯度下降实现
def gradient_descent(X, y, learning_rate=0.1, n_iterations=1000):
    """
    梯度下降求解线性回归
    参数:
        X: 特征矩阵（已包含偏置项）
        y: 目标值
        learning_rate: 学习率
        n_iterations: 迭代次数
    返回:
        theta: 参数 [bias, slope]
        cost_history: 每次迭代的损失值
    """
    m = len(X)  # 样本数
    theta = np.random.randn(2, 1)  # 随机初始化参数
    cost_history = []

    for iteration in range(n_iterations):
        # 前向传播：计算预测值
        y_pred = X @ theta

        # 计算损失（均方误差）
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_history.append(cost)

        # 计算梯度
        gradients = (1 / m) * (X.T @ (y_pred - y))

        # 更新参数
        theta = theta - learning_rate * gradients

        # 每100次打印一次
        if iteration % 200 == 0:
            print(f"迭代 {iteration}, 损失: {cost:.6f}")

    return theta, cost_history


# 5. 训练模型
print("\n开始训练...")
theta, cost_history = gradient_descent(X_b, y, learning_rate=0.1, n_iterations=1000)

print(f"\n训练完成！")
print(f"截距 (bias): {theta[0][0]:.4f}")
print(f"斜率 (slope): {theta[1][0]:.4f}")
print(f"真实值: 截距=4.0, 斜率=3.0")

# 6. 损失曲线
plt.subplot(1, 3, 2)
plt.plot(cost_history)
plt.xlabel("迭代次数")
plt.ylabel("损失值")
plt.title("损失函数下降曲线")
plt.grid(True, alpha=0.3)

# 7. 预测并画拟合直线
X_new = np.array([[0], [2]])  # 取X的最小值和最大值
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta

plt.subplot(1, 3, 3)
plt.scatter(X, y, alpha=0.7, label="真实数据")
plt.plot(X_new, y_predict, 'r-', linewidth=2, label="拟合直线")
plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归拟合结果")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150)
plt.show()

# 8. 模型评估
y_pred_all = X_b @ theta
mse = np.mean((y_pred_all - y) ** 2)
r2 = 1 - (np.sum((y - y_pred_all) ** 2) / np.sum((y - np.mean(y)) ** 2))

print("\n" + "=" * 60)
print("模型评估")
print("=" * 60)
print(f"均方误差 (MSE): {mse:.4f}")
print(f"R² 分数: {r2:.4f}")