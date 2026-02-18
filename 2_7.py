#FPR (False Positive Rate) = FP / (FP+TN) → 误报率
#TPR (True Positive Rate)  = TP / (TP+FN) → 召回率
#roc_auc_score: 计算 AUC 值（ROC 曲线下方的面积）
# ============================================================
# 库导入说明
# ============================================================
import numpy as np              # 数值计算库：高效数组运算、数学函数
import pandas as pd             # 数据处理库：DataFrame 表格操作、CSV 读取
import matplotlib.pyplot as plt # 绑图库：数据可视化
import seaborn as sns           # 高级统计可视化库：更美观的图表封装

# sklearn.model_selection: 模型选择与评估模块
#   train_test_split: 将数据划分为训练集和测试集
#   cross_val_score: K 折交叉验证快捷函数，返回每折的评分数组
#   learning_curve: 学习曲线，用不同大小训练集评估模型表现
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve

# sklearn.preprocessing: 数据预处理模块
#   StandardScaler: 标准化器（均值→0，标准差→1），消除量纲差异
from sklearn.preprocessing import StandardScaler

# sklearn.pipeline: 流水线模块
#   Pipeline: 将预处理和模型打包，交叉验证时防止数据泄露
from sklearn.pipeline import Pipeline

# sklearn.linear_model: 线性模型模块
#   LogisticRegression: 逻辑回归分类器
from sklearn.linear_model import LogisticRegression

# sklearn.neighbors: 近邻模型模块
#   KNeighborsClassifier: K 近邻分类器，根据最近的 K 个样本投票分类
from sklearn.neighbors import KNeighborsClassifier

# sklearn.tree: 决策树模块
#   DecisionTreeClassifier: 决策树分类器，通过特征逐步分裂来分类
from sklearn.tree import DecisionTreeClassifier

# sklearn.ensemble: 集成学习模块
#   RandomForestClassifier: 随机森林，多棵决策树投票（Bagging 思想）
from sklearn.ensemble import RandomForestClassifier

# sklearn.svm: 支持向量机模块
#   SVC: 支持向量分类器，寻找最大间隔的决策边界
from sklearn.svm import SVC

# sklearn.metrics: 评估指标模块
#   roc_curve: 计算不同阈值下的 FPR 和 TPR，用于绘制 ROC 曲线
#   roc_auc_score: 计算 AUC（ROC 曲线下面积），衡量模型排序能力
from sklearn.metrics import roc_curve, roc_auc_score,auc

plt.rcParams["font.sans-serif"] = ["SimHei"]      # 中文字体设置
plt.rcParams["axes.unicode_minus"] = False          # 负号正常显示


# ============================================================
# 数据准备：Titanic 数据加载与清洗（完整版）
# ============================================================

# --- 1. 读取数据 ---
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(f"原始数据形状: {df.shape}")   # (891, 12)

# --- 2. 特征工程 ---
# 从 Name 中提取称谓（Mr/Mrs/Miss 等）
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")

# 合并罕见称谓为 "Rare"
df["Title"] = df["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr",
     "Major","Rev","Sir","Jonkheer","Dona"],
    "Rare"
)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

# 家庭规模 = 兄弟姐妹配偶数 + 父母子女数 + 自己
df["Family_size"] = df["SibSp"] + df["Parch"] + 1

# 是否独自一人
df["Is_alone"] = (df["Family_size"] == 1).astype(int)

# --- 3. 缺失值处理 ---
# Age：按舱位分组，用各组中位数填充（不同舱位年龄分布不同）
df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked：用众数（出现最多的值）填充，只有 2 个缺失
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# --- 4. 编码分类变量 ---
# Sex：二分类，直接映射为 0/1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked 和 Title：多分类，用独热编码
# drop_first=True: 丢弃第一个类别避免多重共线性
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

# --- 5. 删除不用于建模的列 ---
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# --- 6. 分离特征和标签 ---
X = df.drop(columns=["Survived"])    # 特征矩阵
y = df["Survived"]                    # 标签向量

print(f"清洗后数据形状: {df.shape}")
print(f"特征数量: {X.shape[1]}")
print(f"缺失值总数: {df.isnull().sum().sum()}")
print(f"存活率: {y.mean():.3f}")
print(f"\n特征列表: {list(X.columns)}")

# --- 7. 划分训练集/测试集（用于任务2绘制ROC曲线）---
# stratify=y: 分层抽样，保证训练集和测试集的正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
# ============================================================
# 1. 交叉验证对比
#    对以下 5 个模型做 5 折交叉验证，评估 accuracy 和 roc_auc：
#    - LogisticRegression(C=1.0)
#    - KNeighborsClassifier(n_neighbors=5)
#    - DecisionTreeClassifier(max_depth=5)
#    - RandomForestClassifier(n_estimators=100)
#    - SVC(kernel="rbf", probability=True)
#    打印每个模型的 均值±标准差
#    所有模型都要用 Pipeline 包裹 StandardScaler！
pipelines={"逻辑回归":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(C=1.0,random_state=42))]),
           "knn":Pipeline([("scaler",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))]),
           "Deci":Pipeline([("scaler",StandardScaler()),("model",DecisionTreeClassifier(max_depth=5,random_state=42))]),
           "rand":Pipeline([("sclaer",StandardScaler()),("model",RandomForestClassifier(n_estimators=100,random_state=42))]),
           "SVC":Pipeline([("scaler",StandardScaler()),("model",SVC(kernel="rbf",probability=True,random_state=42))]) 
           }
result={}
print("=" * 60)
print(f"{'模型':<12} {'Accuracy':>10} {'AUC':>10} {'F1':>10}")
print("=" * 60)
for name,pipe in pipelines.items():
    acc_scores=cross_val_score(pipe,X,y,cv=5,scoring="accuracy")
    auc_scores=cross_val_score(pipe,X,y,cv=5,scoring="roc_auc")
    f1_scores=cross_val_score(pipe,X,y,cv=5,scoring="f1")
    result[name]={"acc":acc_scores,"auc":auc_scores,"f1":f1_scores}
    print(f"{name:<12} {acc_scores.mean():>8.4f}±{acc_scores.std():.3f}"
          f" {auc_scores.mean():>8.4f}±{auc_scores.std():.3f}"
          f" {f1_scores.mean():>8.4f}±{f1_scores.std():.3f}")

print("=" * 60)
# 2. ROC 曲线对比
#    用 train_test_split 划分数据
#    在同一张图上绘制上述 5 个模型的 ROC 曲线
#    图例中标注每个模型的 AUC 值
#    提示：每个模型都要先 fit，再用 predict_proba 获取概率
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
plt.figure(figsize=(12, 8))
colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E']
line_styles = ['-', '--', '-.', ':', '-']

# 存储AUC值
auc_scores = {}

# 训练和评估每个Pipeline模型
for idx, (name, pipeline) in enumerate(pipelines.items()):
    print(f"训练模型: {name}")
    
    # 训练Pipeline（自动处理所有预处理步骤）
    pipeline.fit(X_train, y_train)
    
    # 直接调用Pipeline的predict_proba
    # Pipeline会自动对测试数据进行相同的预处理
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    auc_scores[name] = roc_auc
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color=colors[idx], 
             linestyle=line_styles[idx],
             lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
    
    # 添加填充
    plt.fill_between(fpr, tpr, alpha=0.1, color=colors[idx])

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier (AUC = 0.5)')

# 设置图表属性
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13)
plt.title('ROC Curves Comparison - Models with Pipeline', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.show()

# 打印结果
print("\n模型性能排名 (按AUC):")
for i, (name, auc_score) in enumerate(sorted(auc_scores.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True), 1):
    print(f"{i}. {name}: {auc_score:.4f}")
# 3. 学习曲线
#    选出交叉验证中表现最好的模型
#    绘制它的学习曲线
#    判断：它是过拟合还是欠拟合？有没有改进空间？
def plot_learning_curve(estimator, title, X, y, ax, cv=5):
    """
    绘制学习曲线，诊断过拟合/欠拟合

    参数:
        estimator: 模型（Pipeline 或单个模型）
        title: 图表标题
        X, y: 数据
        ax: matplotlib 的 Axes 对象（用于在子图上绘制）
        cv: 交叉验证折数
    """
    # np.linspace(0.1, 1.0, 10): 从 10% 到 100% 的训练集大小
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        n_jobs=-1     # n_jobs=-1: 使用所有 CPU 核心并行计算，加速
    )

    # 计算每个训练集大小下的均值和标准差
    train_mean = train_scores.mean(axis=1)     # axis=1: 对每行（每个大小）求均值
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # 绘制均值曲线
    ax.plot(train_sizes, train_mean, "o-", color="#2196F3", label="训练集分数", linewidth=2)
    ax.plot(train_sizes, test_mean, "o-", color="#FF5722", label="验证集分数", linewidth=2)

    # fill_between(): 绘制±1标准差的阴影区域，表示分数的波动范围
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="#2196F3")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.1, color="#FF5722")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("训练集大小")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 标注最终的训练/验证差距
    gap = train_mean[-1] - test_mean[-1]
    ax.annotate(f"差距={gap:.3f}", xy=(train_sizes[-1], test_mean[-1]),
                xytext=(train_sizes[-1]-200, test_mean[-1]-0.05),
                fontsize=10, arrowprops=dict(arrowstyle="->"))

# ============================================================
# 对比：欠拟合 vs 正常 vs 过拟合
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
pipebest=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(C=1.0,random_state=42))])
plot_learning_curve(pipebest, "逻辑回归", X, y, axes[2])

plt.suptitle("学习曲线 — 诊断过拟合/欠拟合", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# 4. 回答问题（写在注释中）：
#    (a) 哪个模型的交叉验证分数最高？哪个最稳定（标准差最小）？
#    (b) 看 ROC 曲线，哪个模型的 AUC 最高？
#    (c) 如果训练集分数是 0.95 但验证集分数是 0.78，这是什么问题？该怎么办？