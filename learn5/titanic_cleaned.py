import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
import matplotlib.pyplot as plt

print('=' * 60)
print('scikit-learn 基础教程')
print('=' * 60)

#数据准备
#生成示例数据
np.random.seed(42)
X = np.random.randn(200,3)
y = 2*X[:,0] + 3*X[:,1] - 1.5*X[:,2] + np.random.randn(200) * 0.1
#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print(f"训练集大小:{X_train.shape}")
print(f"测试集大小:{X_test.shape}")
print(f"训练集目标值范围:{y_train.min():.2f},{y_train.max():.2f}")

#特征缩放
#标准化(均值为0，标准差为1)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
#归一化（缩放到[0,1]区间）
scaler_minmax = MinMaxScaler()
X_train_mm  = scaler_minmax.fit_transform(X_train)
X_test_mm = scaler_minmax.transform(X_test)
print(f"标准化前 - 训练集均值:{X_train.mean():.3f},标准差：{X_train.std():.3f}")
print(f"标准化后 - 训练集均值:{X_train_std.mean():.3f},标准差：{X_train_std.std():.3f}")
print(f"归一化后 - 训练集范围:{X_train_mm.min():.3f},{X_train_mm.max():.3f}")

#线性回归
#训练模型
lr = LinearRegression()
lr.fit(X_train_std,y_train)
#预测
y_train_pred = lr.predict(X_train_std)
y_test_pred = lr.predict(X_test_std)
#评估
train_mse = mean_squared_error(y_train,y_train_pred)
test_mse = mean_squared_error(y_test,y_test_pred)
train_r2 = r2_score(y_train,y_train_pred)
test_r2 = r2_score(y_test,y_test_pred)

print(f"训练集 MSE:{train_mse:.4f},R²:{train_r2:.4f}")
print(f"测试集 MSE：{test_mse:.4f},R²：{test_r2:.4f}")
print(f"模型系数：{lr.coef_}")
print(f"模型截距：{lr.intercept_:.4f}")
#可视化预测结果
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(y_train,y_train_pred,alpha=0.6)
plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],
         'r--',linewidth=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Linear Regression: Predictions vs True Values')


plt.subplot(1,2,2)
residuals = y_test - y_test_pred
plt.hist(residuals,bins=50,edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.axvline(0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\linear_regression_demo.png', dpi=150)
plt.show()

#逻辑回归
#生成二分类数据
X_clf = np.random.randn(300,2)
y_clf = (X_clf[:,0] + X_clf[:,1]>0).astype(int)
X_train_clf,X_test_clf,y_train_clf,y_test_clf = train_test_split(
    X_clf,y_clf,test_size=0.3,random_state=42
)
#训练逻辑回归
log_reg = LogisticRegression()
log_reg.fit(X_train_clf,y_train_clf)
#预测
y_pred_clf = log_reg.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf,y_pred_clf)
print(f"逻辑回归准确率:{accuracy:.4f}")

#可视化决策边界
plt.figure(figsize=(8,6))
#绘制决策边界
x_min,x_max = X_clf[:,0].min()-0.5,X_clf[:,0].max()+0.5
y_min,y_max = X_clf[:,1].min()-0.5,X_clf[:,1].max()+0.5
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),
                    np.arange(y_min,y_max,0.02))
Z = log_reg.predict(np.c_[xx.ravel(),np.ravel(yy)])
Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.3,cmap='coolwarm')
plt.scatter(X_test_clf[:,0],X_test_clf[:,1],
            c=y_test_clf,cmap='coolwarm',edgecolors='black')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Logistic Regression Decision Boundary (Accuracy: {accuracy:.3f})')
plt.savefig(r'E:\python project\ai-learning\learn5\logistic_regression_demo.png', dpi=150)
plt.show()