import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("=" * 60)
print("Titanic 数据预处理")
print("=" * 60)

df = pd.read_csv(r'E:\python project\ai-learning\learn3\data\titanic_cleaned.csv')
print("从 learn3/titanic_cleaned.csv 加载数据")

print(f'数据形状:{df.shape}')
print(f"列名:{df.columns.tolist()}")
print(f"前五行:{df.head()}")

#分离特征和标签
X = df.drop('Survived', axis=1)
y = df['Survived']

print(f"特征形状:{X.shape}")
print(f"目标形状:{y.shape}")
print(f"目标分布:\n{y.value_counts()}")
print(f"存活率:{y.mean():.3f}")

#处理分类变量
#查看分类别
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"分类别:{categorical_cols}")
#标签编码
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"{col}:{dict(zip(le.classes_, le.transform(le.classes_)))}")
#处理数值特征
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"数值列:{numeric_cols}")
print(f"\n数值特征统计:\n{X[numeric_cols].describe()}")
#特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"缩放后特征均值:{X_scaled.mean():.3f}")
print(f"缩放后特征标准差:{X_scaled.std():.3f}")
#数据集划分
X_train_scale,X_test_scale,y_train_scale,y_test_scale = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42,stratify=y)
print(f"训练集大小：{X_train_scale.shape}")
print(f"测试集大小：{y_train_scale.shape}")
print(f"训练集存活率:{X_test_scale.mean():.3f}")
print(f"测试集存活率:{y_test_scale.mean():.3f}")

#可视化数据分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 年龄分布
axes[0, 0].hist(df[df['Survived']==1]['Age'], bins=20, alpha=0.7, label='Survived')
axes[0, 0].hist(df[df['Survived']==0]['Age'], bins=20, alpha=0.7, label='Died')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution by Survival')
axes[0, 0].legend()

# 票价分布
axes[0, 1].hist(df[df['Survived']==1]['Fare'], bins=20, alpha=0.7, label='Survived')
axes[0, 1].hist(df[df['Survived']==0]['Fare'], bins=20, alpha=0.7, label='Died')
axes[0, 1].set_xlabel('Fare')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Fare Distribution by Survival')
axes[0, 1].legend()

# 舱位分布
pclass_survival = df.groupby('Pclass')['Survived'].mean()
axes[0, 2].bar(pclass_survival.index, pclass_survival.values)
axes[0, 2].set_xlabel('Pclass')
axes[0, 2].set_ylabel('Survival Rate')
axes[0, 2].set_title('Survival Rate by Pclass')

# 性别分布
gender_survival = df.groupby('Sex')['Survived'].mean()
axes[1, 0].bar(['Male', 'Female'], gender_survival.values)
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Survival Rate')
axes[1, 0].set_title('Survival Rate by Gender')

# 家庭大小分布
family_survival = df.groupby('FamilySize')['Survived'].mean()
axes[1, 1].plot(family_survival.index, family_survival.values, marker='o')
axes[1, 1].set_xlabel('Family Size')
axes[1, 1].set_ylabel('Survival Rate')
axes[1, 1].set_title('Survival Rate by Family Size')
axes[1, 1].grid(True, alpha=0.3)

# 相关性热力图
numeric_cols_all = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Survived']
corr = df[numeric_cols_all].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, ax=axes[1, 2])
axes[1, 2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\titanic_distributions.png', dpi=150)
plt.show()

# 8. 保存预处理后的数据

joblib.dump(scaler, r'E:\python project\ai-learning\learn5\scaler.pkl')
for col, le in label_encoders.items():
    joblib.dump(le, r'E:\python project\ai-learning\learn5\label_encoder_{col}.pkl')

# 保存划分的数据
np.savez(r'E:\python project\ai-learning\learn5\titanic_split.npz',
         X_train=X_train_scale, X_test=X_test_scale,
         y_train=y_train_scale, y_test=y_test_scale)

print("\n数据预处理完成")
print("保存的文件:")
print("  - learn5/scaler.pkl")
print("  - learn5/label_encoder_*.pkl")
print("  - learn5/titanic_split.npz")
print("  - learn5/titanic_distributions.png")

