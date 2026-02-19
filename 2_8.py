# ============================================================
# 库导入
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 数据准备
# ============================================================
digits = load_digits()
X = digits.data       # (1797, 64)
y = digits.target     # (1797,) 0~9

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 任务清单
# ============================================================

# 1. 核函数对比
#    用 Pipeline(StandardScaler + SVC) 对比以下 4 种核函数的 5 折 CV 准确率：
#      - linear (C=1.0)
#      - rbf (C=1.0, gamma="scale")
#      - rbf (C=10.0, gamma="scale")
#      - poly (C=1.0, degree=3)
#    打印每种的 均值±标准差，找出最佳核函数
kernals={"linear":SVC(kernel="linear",C=1.0,random_state=42),
         "rbf1.0":SVC(kernel="rbf",C=1.0,gamma="scale",random_state=42),
         "rbf10.0":SVC(kernel="rbf",C=10.0,gamma="scale",random_state=42),
         "poly":SVC(kernel="poly",C=1.0,degree=3)}
best_score = 0
best_name = ""
for name,model in kernals.items():
    pipe=Pipeline([("scaler",StandardScaler()),
                  ("svm",model)])
    scores=cross_val_score(pipe,X,y,cv=5,scoring="accuracy")
    print(f"{name:<22} {scores.mean():.4f}±{scores.std():.3f}")
    if scores.mean()>best_score:
        best_score=scores.mean()
        best_name=name 
print("=" * 55)
print(f"🏆 最佳核函数: {best_name} ({best_score:.4f})")
# 2. C 和 gamma 调参（网格搜索思想）
#    对 RBF 核，遍历以下参数组合，记录每组的 CV 准确率：
#      C: [0.1, 1, 10, 100]
#      gamma: [0.001, 0.01, 0.1, 1]
#    找出最佳 (C, gamma) 组合
#    提示：用两层 for 循环即可
C_test=[0.1, 1, 10, 100]
gamma_test=[0.001, 0.01, 0.1, 1]
C_best=0
gamma_best=0
rbf_score2=0
for C in C_test:
    for gamma in gamma_test:
        rbf_model=Pipeline([("scaler",StandardScaler()),
                     ("rbf",SVC(kernel="rbf",C=C,gamma=gamma,random_state=42))])
        rbf_model.fit(X_train, y_train)
        y_pred = rbf_model.predict(X_test)
        score2=cross_val_score(rbf_model,X,y,cv=5,scoring="accuracy")
        print(f"{C:<11}{gamma:<11} {score2.mean():.4f}±{score2.std():.3f}")
        if score2.mean() > rbf_score2:
            rbf_score2=score2.mean()
            C_best=C
            gamma_best=gamma

print(f"\n最佳模型测试集准确率: {rbf_score2:.4f}最佳（C，gamma）组合：{C_best,gamma_best}")
# 3. 用最佳参数训练模型
#    在测试集上评估，打印 classification_report
#    绘制 10×10 的混淆矩阵热力图
best_model=Pipeline([("scaler",StandardScaler()),
                     ("rbf",SVC(kernel="rbf",C=C_best,gamma=gamma_best,random_state=42))])
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
score3=cross_val_score(best_model,X,y,cv=5,scoring="accuracy")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图1：混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=range(10), yticklabels=range(10))
axes[0].set_title("混淆矩阵 — 手写数字识别", fontsize=13)
axes[0].set_xlabel("预测数字")
axes[0].set_ylabel("真实数字")

# 子图2：显示分错的样本
# np.where(条件): 返回满足条件的索引
wrong_idx = np.where(y_pred != y_test)[0]    # 找出预测错误的样本索引

if len(wrong_idx) > 0:
    # 最多展示 10 个错误样本
    show_count = min(10, len(wrong_idx))
    for i in range(show_count):
        idx = wrong_idx[i]

        # 在子图2中创建小图
        # add_subplot(rows, cols, index): 在 axes[1] 的区域内创建子图
        ax_small = fig.add_axes([0.55 + (i % 5) * 0.09, 0.55 - (i // 5) * 0.45,
                                  0.07, 0.35])
        ax_small.imshow(X_test[idx].reshape(8, 8), cmap="gray_r")
        ax_small.set_title(f"真:{y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]}"
                          f"→预:{y_pred[idx]}", fontsize=8, color="red")
        ax_small.axis("off")

    axes[1].set_title(f"预测错误的样本 ({len(wrong_idx)}个)", fontsize=13)
    axes[1].axis("off")
else:
    axes[1].text(0.5, 0.5, "没有预测错误！🎉", fontsize=20,
                ha="center", va="center", transform=axes[1].transAxes)
    axes[1].axis("off")

plt.tight_layout()
plt.show()
# 4. 错误分析
#    找出所有预测错误的样本
#    可视化其中 10 个（显示真实标签和预测标签）
#    分析：哪些数字最容易被混淆？（看混淆矩阵的非对角线元素）
#1与4，7与9，6与9，4与8，1与8，4与7
# 5. 回答问题（写在注释中）：
#    (a) RBF 核为什么通常比线性核好？
#将向量映射到无限维，线性核只能在二维操作
#    (b) 增大 C 和增大 gamma 对模型复杂度的影响方向是否相同？
#是
#    (c) SVM 适合处理这个手写数字识别问题吗？它有什么局限性？
#适合，但训练慢，参数敏感