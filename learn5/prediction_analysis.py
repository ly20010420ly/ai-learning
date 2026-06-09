import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from learn5.titanic_models import y_pred_proba

print('=' * 60)
print('模型预测和结果分析')
print('=' * 60)

#加载最佳模型
best_model = joblib.load(r'E:\python project\ai-learning\learn5\optimized_titanic_model.pkl')
df = pd.read_csv(r'E:\python project\ai-learning\learn3\data\titanic_cleaned.csv')

#编码分类变量
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:,1]

#创建结果Dataframe
results_df = X_test.copy()
results_df['True_Survived'] = y_test.values
results_df['Predicted_Survived'] = y_pred
results_df['Survival_Probability'] = y_pred_proba
results_df['Correct'] = (y_test == y_pred).values

print(f"\n预测结果统计:")
print(f"总样本数: {len(results_df)}")
print(f"正确预测数: {results_df['Correct'].sum()}")
print(f"准确率: {results_df['Correct'].mean():.4f}")

#错误分析
# 找出错误预测的样本
false_positives = results_df[(results_df['True_Survived'] == 0) & (results_df['Predicted_Survived'] == 1)]
false_negatives = results_df[(results_df['True_Survived'] == 1) & (results_df['Predicted_Survived'] == 0)]
print(f"假阳性（预测存活实际死亡）: {len(false_positives)} 个")
print(f"假阴性（预测死亡实际存活）: {len(false_negatives)} 个")

#  可视化错误类型
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 预测概率分布
axes[0, 0].hist(results_df[results_df['True_Survived']==0]['Survival_Probability'],
                 bins=20, alpha=0.7, label='Actual Died', color='red')
axes[0, 0].hist(results_df[results_df['True_Survived']==1]['Survival_Probability'],
                 bins=20, alpha=0.7, label='Actual Survived', color='green')
axes[0, 0].set_xlabel('Survival Probability')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Prediction Probability Distribution')
axes[0, 0].legend()

# 5.2 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'], ax=axes[0, 1])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix')

# 5.3 错误样本的特征分布
if len(false_positives) > 0:
    axes[1, 0].scatter(false_positives['Age'], false_positives['Fare'],
                       c='red', label='False Positive', alpha=0.6, s=50)
if len(false_negatives) > 0:
    axes[1, 0].scatter(false_negatives['Age'], false_negatives['Fare'],
                       c='blue', label='False Negative', alpha=0.6, s=50)
axes[1, 0].scatter(results_df[results_df['Correct']]['Age'],
                   results_df[results_df['Correct']]['Fare'],
                   c='green', label='Correct', alpha=0.3, s=30)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Fare')
axes[1, 0].set_title('Error Analysis: Age vs Fare')
axes[1, 0].legend()

# 5.4 舱位错误率分析
pclass_error = results_df.groupby('Pclass')['Correct'].apply(lambda x: (1 - x.mean()))
axes[1, 1].bar(pclass_error.index, pclass_error.values)
axes[1, 1].set_xlabel('Passenger Class')
axes[1, 1].set_ylabel('Error Rate')
axes[1, 1].set_title('Error Rate by Passenger Class')
axes[1, 1].set_xticks([1, 2, 3])

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\prediction_analysis.png', dpi=150)
plt.show()

# 6. 特征对预测的影响
print("\n5. 特征影响分析...")

# 分析不同特征的预测分布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 性别影响
gender_cols = [c for c in results_df.columns if 'Sex' in c]
if gender_cols:
    gender_data = results_df.groupby(gender_cols[0])['Survival_Probability'].mean()
    axes[0, 0].bar(['Male', 'Female'], gender_data.values)
    axes[0, 0].set_xlabel('Gender')
    axes[0, 0].set_ylabel('Avg Survival Probability')
    axes[0, 0].set_title('Impact of Gender on Predictions')

# 舱位影响
pclass_proba = results_df.groupby('Pclass')['Survival_Probability'].mean()
axes[0, 1].bar(pclass_proba.index, pclass_proba.values)
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylabel('Avg Survival Probability')
axes[0, 1].set_title('Impact of Pclass on Predictions')

# 年龄影响
age_bins = pd.cut(results_df['Age'], bins=5)
age_proba = results_df.groupby(age_bins)['Survival_Probability'].mean()
axes[1, 0].plot(range(len(age_proba)), age_proba.values, marker='o')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Avg Survival Probability')
axes[1, 0].set_title('Impact of Age on Predictions')
axes[1, 0].set_xticks(range(len(age_proba)))
axes[1, 0].set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in age_proba.index], rotation=45)

# 家庭大小影响
family_proba = results_df.groupby('FamilySize')['Survival_Probability'].mean()
axes[1, 1].plot(family_proba.index, family_proba.values, marker='o')
axes[1, 1].set_xlabel('Family Size')
axes[1, 1].set_ylabel('Avg Survival Probability')
axes[1, 1].set_title('Impact of Family Size on Predictions')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\feature_impact.png', dpi=150)
plt.show()

# 7. 生成最终报告
print("\n" + "=" * 60)
print("最终分析报告")
print("=" * 60)

report = f"""
Titanic生存预测 - 模型分析报告

1. 模型性能:
   - 准确率: {results_df['Correct'].mean():.4f}
   - 正确预测: {results_df['Correct'].sum()}/{len(results_df)}
   - 假阳性: {len(false_positives)} (预测存活实际死亡)
   - 假阴性: {len(false_negatives)} (预测死亡实际存活)

2. 重要特征:
   - 性别是最重要的预测因素
   - 舱位等级影响显著
   - 家庭大小适中（2-4人）预测存活率更高

3. 模型局限性:
   - 对极端年龄的预测可能不准
   - 部分假阳性来自三等舱
   - 需要更多特征来提高准确率

4. 改进建议:
   - 添加更多特征（如家庭成员的具体关系）
   - 尝试集成学习方法
   - 收集更多训练数据
"""

print(report)

# 保存预测结果
results_df.to_csv(r'E:\python project\ai-learning\learn5\predictions_results.csv', index=False)
print("\n预测结果已保存: E:\python project\ai-learning\learn5\predictions_results.csv")

print("\n 模型预测和分析完成")


