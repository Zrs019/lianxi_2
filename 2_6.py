# ============================================================
# 库导入说明
# ============================================================
import numpy as np              # 数值计算库：高效数组运算
import pandas as pd             # 数据处理库：DataFrame 操作、CSV 读取等
import matplotlib.pyplot as plt # 绑图库：数据可视化
import seaborn as sns           # 统计可视化库：基于 matplotlib 的高级封装，图表更美观

# sklearn.model_selection: 模型选择模块
#   train_test_split: 将数据按比例划分为训练集和测试集
from sklearn.model_selection import train_test_split

# sklearn.preprocessing: 数据预处理模块
#   StandardScaler: 标准化（均值变0、标准差变1），消除特征量纲差异
from sklearn.preprocessing import StandardScaler

# sklearn.linear_model: 线性模型模块
#   LogisticRegression: 逻辑回归分类器
#     虽然名字叫"回归"，但它是分类算法！
#     通过 Sigmoid 函数将线性输出转为概率
from sklearn.linear_model import LogisticRegression

# sklearn.metrics: 模型评估指标模块
#   accuracy_score: 准确率 = 预测正确数 / 总数
#   classification_report: 生成包含 precision/recall/f1 的完整报告
#   confusion_matrix: 混淆矩阵（TP/TN/FP/FN）
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 1. 数据加载与清洗
# ============================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# pd.read_csv(url): 从 URL 或文件路径读取 CSV 文件为 DataFrame
df = pd.read_csv(url)
print(f"原始数据形状: {df.shape}")    # (891, 12)

# ---------- 特征工程 ----------

# .str.extract(正则表达式): 从字符串中提取匹配的部分
# r" ([A-Za-z]+)\.": 匹配空格后、点号前的英文单词（即称谓）
# 例如 "Braund, Mr. Owen Harris" → 提取 "Mr"
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")

# .replace(字典): 将指定值替换为新值
# 把罕见称谓合并为 "Rare"，把 Mlle/Ms 归为 Miss，Mme 归为 Mrs
df["Title"] = df["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],
    "Rare"
)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

# 创建家庭规模特征
df["Family_size"] = df["SibSp"] + df["Parch"] + 1

# 创建是否独自一人的特征
# .astype(int): 将布尔值转为整数（True→1, False→0）
df["Is_alone"] = (df["Family_size"] == 1).astype(int)

# ---------- 缺失值处理 ----------

# .groupby("col")["target"].transform(func):
#   按分组计算，结果保持原始 DataFrame 的形状
#   这里按 Pclass 分组，用各组的中位数填充 Age 的缺失值
#   比用全局中位数更精确（不同舱位的乘客年龄分布不同）
df["Age"] = df.groupby("Pclass")["Age"].transform(
    lambda x: x.fillna(x.median())    # lambda 匿名函数：对每组执行 fillna
)

# .mode()[0]: 求众数（出现最多的值），[0]取第一个
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# ---------- 编码分类变量 ----------

# .map(字典): 将列中的值按字典映射为新值
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# pd.get_dummies(df, columns=[列名], drop_first=True):
#   独热编码（One-Hot Encoding）
#   将分类变量展开为多个 0/1 列
#   drop_first=True: 丢弃第一个类别，避免多重共线性
#   例如 Embarked 有 S/C/Q → 生成 Embarked_Q, Embarked_S 两列
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

# ---------- 选择特征 ----------

# .drop(columns=[列名列表]): 删除指定列
# 删除不用于建模的列（ID、名字、票号、船舱号）
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
df = df.drop(columns=drop_cols)

print(f"清洗后数据形状: {df.shape}")
print(f"缺失值: {df.isnull().sum().sum()}")
print(f"\n最终特征列表:\n{list(df.columns)}")

# ============================================================
# 2. 划分特征和标签
# ============================================================
# X: 特征矩阵（所有列除了 Survived）
# y: 标签向量（Survived 列）
X = df.drop(columns=["Survived"])
y = df["Survived"]

# train_test_split(): 划分训练集和测试集
#   test_size=0.2: 20% 作为测试集
#   random_state=42: 随机种子，保证可复现
#   stratify=y: 分层抽样，保证训练集和测试集中正负样本比例一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集存活率: {y_train.mean():.3f}")
print(f"测试集存活率: {y_test.mean():.3f}")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. 训练逻辑回归模型
# ============================================================
# LogisticRegression() 参数说明：
#   C=1.0: 正则化强度的倒数（C越大→正则化越弱→模型越复杂）
#          注意：和 Ridge/Lasso 的 alpha 相反！C = 1/alpha
#   penalty="l2": 正则化类型，默认 L2（和 Ridge 一样）
#   max_iter=1000: 最大迭代次数（复杂数据需增大，避免收敛警告）
#   solver="lbfgs": 优化算法，默认 lbfgs（适合中小数据集）
#   random_state=42: 随机种子
C_test=[0.001, 0.01, 0.1, 1, 10, 100]
models=[]
for i in C_test:
    models[i]=LogisticRegression(
    C=C_test[i],
    penalty="l2",
    max_iter=1000,
    random_state=42
)
print(models)
# fit(X, y): 训练模型，从数据中学习权重 w 和截距 b
model.fit(X_train_scaled, y_train)

# ============================================================
# 4. 预测
# ============================================================
# predict(X): 直接预测类别（0 或 1）
#   内部流程：计算概率 → 和阈值0.5比较 → 输出类别
y_pred = model.predict(X_test_scaled)

# predict_proba(X): 预测每个类别的概率
#   返回 shape: (n_samples, n_classes)
#   例如 [[0.85, 0.15],  → 85% 概率是类别0，15% 概率是类别1
#         [0.30, 0.70]]  → 30% 概率是类别0，70% 概率是类别1
y_proba = model.predict_proba(X_test_scaled)

print(f"\n前5个样本的预测概率:")
print(f"{'遇难概率':>10} {'存活概率':>10} {'预测':>6} {'真实':>6}")
print("-" * 40)
for i in range(5):
    print(f"{y_proba[i, 0]:>10.4f} {y_proba[i, 1]:>10.4f} {y_pred[i]:>6} {y_test.iloc[i]:>6}")

# ============================================================
# 5. 模型评估
# ============================================================
# accuracy_score(y_true, y_pred): 准确率
#   = 预测正确的样本数 / 总样本数
acc = accuracy_score(y_test, y_pred)
print(f"\n准确率: {acc:.4f}")

# classification_report(y_true, y_pred): 生成完整的分类报告
#   包含每个类别的 precision, recall, f1-score, support
#   precision（精确率）: 预测为正的样本中，真正为正的比例 → TP/(TP+FP)
#   recall（召回率）:    真正为正的样本中，被预测为正的比例 → TP/(TP+FN)
#   f1-score:           precision 和 recall 的调和平均
#   support:            每个类别的样本数
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=["遇难", "存活"]))

# confusion_matrix(y_true, y_pred): 混淆矩阵
#   [[TN, FP],    TN=真负（预测遇难，实际也遇难）  FP=假正（预测存活，实际遇难）
#    [FN, TP]]    FN=假负（预测遇难，实际存活）    TP=真正（预测存活，实际也存活）
cm = confusion_matrix(y_test, y_pred)

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- 子图1：混淆矩阵热力图 ---
# sns.heatmap(): 绘制热力图
#   annot=True: 在格子里显示数字
#   fmt="d": 数字格式为整数
#   cmap="Blues": 蓝色色系
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["遇难", "存活"], yticklabels=["遇难", "存活"])
axes[0].set_title("混淆矩阵", fontsize=14)
axes[0].set_xlabel("预测值")
axes[0].set_ylabel("真实值")

# --- 子图2：特征重要性（权重绝对值）---
# model.coef_[0]: 逻辑回归的权重数组
#   注意：逻辑回归的 coef_ 是二维的 shape: (n_classes, n_features)
#   二分类时取 [0] 即可
feature_importance = pd.Series(
    np.abs(model.coef_[0]),       # np.abs(): 取绝对值
    index=X.columns               # 用特征名作为索引
).sort_values(ascending=True)     # 按值升序排列（barh 从下到上画）

feature_importance.plot(kind="barh", ax=axes[1], color="#2196F3")
axes[1].set_title("特征重要性（|权重|）", fontsize=14)
axes[1].set_xlabel("权重绝对值")

# --- 子图3：预测概率分布 ---
# 看模型的预测概率分布，理想情况是两类明显分开
axes[2].hist(y_proba[y_test == 0, 1], bins=25, alpha=0.6,
             color="#f44336", label="实际遇难", edgecolor="white")
axes[2].hist(y_proba[y_test == 1, 1], bins=25, alpha=0.6,
             color="#4CAF50", label="实际存活", edgecolor="white")
axes[2].axvline(x=0.5, color="black", linestyle="--", label="决策边界(0.5)")
axes[2].set_title("预测概率分布", fontsize=14)
axes[2].set_xlabel("预测为存活的概率")
axes[2].set_ylabel("样本数")
axes[2].legend()

plt.suptitle("逻辑回归 — Titanic 生存预测", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# 请基于第四节的完整代码，继续完成以下任务
# ============================================================

# 1. 调参实验：测试 C = [0.001, 0.01, 0.1, 1, 10, 100] 
#    打印每个 C 值对应的训练/测试准确率
#    找出最佳 C 值

# 2. 用最佳 C 值重新训练模型
#    打印 classification_report
#    绘制混淆矩阵

# 3. 用 predict_proba() 获取预测概率
#    找出模型"最不确定"的5个样本（概率最接近0.5的）
#    打印它们的特征、预测概率和真实标签
#    提示：np.abs(y_proba[:, 1] - 0.5) 求和 0.5 的距离

# 4. 分析特征重要性：
#    哪3个特征对"存活"影响最大？（看 model.coef_ 的绝对值）
#    它们的权重是正还是负？正负意味着什么？

# 5. 回答问题（写在注释中）：
#    (a) 为什么 Sex（性别）的权重最大？
#    (b) 如果把阈值从 0.5 改为 0.3，准确率会怎样变化？Recall 呢？
#    (c) 逻辑回归适合处理 Titanic 这个问题吗？为什么？