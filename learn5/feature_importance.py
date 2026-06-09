import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("特征重要性分析和模型优化")
print("=" * 60)

df = pd.read_csv(r'E:\python project\ai-learning\learn3\data\titanic_cleaned.csv')

# 编码分类变量
from sklearn.preprocessing import LabelEncoder
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Survived', axis=1)
y = df['Survived']

feature_names = X.columns.tolist()
print(f"特征数量: {len(feature_names)}")
print(f"特征名称: {feature_names}")

# 2. 随机森林特征重要性
print("\n2. 随机森林特征重要性分析...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\feature_importance_rf.png', dpi=150)
plt.show()

print("\n特征重要性排名:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# 3. 特征选择
print("\n3. 特征选择...")

# 单变量特征选择
selector = SelectKBest(f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_mask = selector.get_support()
selected_features = [feature_names[i] for i, mask in enumerate(selected_mask) if mask]

print(f"选择的前5个特征: {selected_features}")

# 比较使用全部特征和选择后的特征
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_all = RandomForestClassifier(random_state=42)
rf_selected = RandomForestClassifier(random_state=42)

# 转换特征
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

rf_all.fit(X_train, y_train)
rf_selected.fit(X_train_selected, y_train)

score_all = rf_all.score(X_test, y_test)
score_selected = rf_selected.score(X_test_selected, y_test)

print(f"\n使用全部特征 ({len(feature_names)}个): {score_all:.4f}")
print(f"使用选择的特征 ({len(selected_features)}个): {score_selected:.4f}")

# 4. 超参数调优
print("\n4. 超参数调优...")

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("参数网格:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# 网格搜索
rf_tune = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_tune, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

print("\n开始网格搜索...")
grid_search.fit(X_train, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 评估最佳模型
best_rf = grid_search.best_estimator_
test_score = best_rf.score(X_test, y_test)
print(f"测试集分数: {test_score:.4f}")

# 5. 优化后的特征重要性
print("\n5. 优化后的特征重要性...")

optimized_importances = best_rf.feature_importances_
optimized_indices = np.argsort(optimized_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(optimized_importances)), optimized_importances[optimized_indices])
plt.xticks(range(len(optimized_importances)),
           [feature_names[i] for i in optimized_indices],
           rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Optimized Random Forest - Feature Importance')
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\optimized_feature_importance.png', dpi=150)
plt.show()

# 6. 学习曲线分析
print("\n6. 学习曲线分析...")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, test_mean, 'o-', label='Cross-Validation Score', linewidth=2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\python project\ai-learning\learn5\learning_curves.png', dpi=150)
plt.show()

# 7. 交叉验证详细结果
print("\n7. 交叉验证详细结果...")
cv_scores = cross_val_score(best_rf, X, y, cv=10)
print(f"10折交叉验证分数: {cv_scores}")
print(f"平均分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 保存优化后的模型
import joblib
joblib.dump(best_rf, r'E:\python project\ai-learning\learn5\optimized_titanic_model.pkl')
joblib.dump(selector, r'E:\python project\ai-learning\learn5\feature_selector.pkl')

print("\n 特征分析和模型优化完成")
print("保存的文件:")
print("  - learn5/optimized_titanic_model.pkl")
print("  - learn5/feature_selector.pkl")
print("  - learn5/feature_importance_rf.png")
print("  - learn5/optimized_feature_importance.png")
print("  - learn5/learning_curves.png")