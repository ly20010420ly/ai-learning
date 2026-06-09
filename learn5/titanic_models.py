import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from sklearn.linear_model import LogisticRegression  #逻辑回归
from sklearn.tree import DecisionTreeClassifier     #决策树分类器
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier  #随机森林、梯度提升树
from sklearn.svm import SVC                         #支持向量机分类器
from sklearn.neighbors import KNeighborsClassifier  #K近邻分类器
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  #准确率、精确率、召回率、f1分数
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc
                            #显示TP、FP、TN、FN数量，直观了解分类效果  #包含精确率、召回率、F1分数和样本数的综合报告  #用于二分类模型性能评估，AUC值越大表示模型越好
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from learn5.titanic_preprocessing import fig

warnings.filterwarnings("ignore")

print("=" * 60)
print("Titanic 多模型训练和对比")
print("=" * 60)

#加载预处理数据
data = np.load(r'E:\python project\ai-learning\learn5\titanic_split.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(f"训练集：{X_train.shape}")
print(f"测试集：{X_test.shape}")

#定义多个模型
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree':DecisionTreeClassifier(random_state=42),
    'Random Forest':RandomForestClassifier(random_state=42),  #随机森林
    'Gradient Bossting':GradientBoostingClassifier(random_state=42),
    'SVM':SVC(random_state=42,probability=True),
    'KNN':KNeighborsClassifier(),
}
#训练和评估
results = []
for name,model in models.items():
    print(f"\n训练{name}...")
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    #计算指标
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    results.append({
        "model":name,
        "Accuracy":accuracy,
        "Precision":precision,
        "Recall":recall,
        "F1-Score":f1,
    })

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

#结果对比
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Accuracy',ascending=False)
print(results_df.to_string(index=False))

#可视化模型对比
fig,axes = plt.subplots(2,2,figsize=(14,10))
#柱状图对比
metrics = ['Accuracy','Precision','Recall','F1-Score']
for i,metric in enumerate(metrics):
    ax = axes[i//2,i%2]
    bars = ax.bar(results_df['model'],results_df[metric])
    ax.set_ylim([0,1])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.set_xticklabels(results_df['model'],rotation=45,ha='right')
    #添加数值标签
    for bar, value in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\models_comparison.png', dpi=150)
plt.show()

#最佳模型详细分析
best_model_name = results_df.iloc[0]['model']
best_model = models[best_model_name]
print(f"最佳模型:{best_model_name}")
#预测
y_pred_best = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:,1]
#混淆矩阵
cm = confusion_matrix(y_test,y_pred_best)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{best_model_name} - Confusion Matrix')

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{best_model_name} - ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\python project\ai-learning\learn5\best_model_analysis.png', dpi=150)
plt.show()

# 分类报告
print(f"\n分类报告:\n{classification_report(y_test, y_pred_best)}")

# 保存最佳模型
import joblib
joblib.dump(best_model, r'E:\python project\ai-learning\learn5\best_titanic_model.pkl')
print(f"\n最佳模型已保存: E:\python project\ai-learning\learn5\best_titanic_model.pkl")

print("\n 模型训练和对比完成")
