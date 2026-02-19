# ============================================================
# 库导入
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 数据准备（Titanic 完整清洗流程）
# ============================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
df["Title"] = df["Title"].replace(
    ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"], "Rare"
)
df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")
df["Family_size"] = df["SibSp"] + df["Parch"] + 1
df["Is_alone"] = (df["Family_size"] == 1).astype(int)
df["Age"] = df.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 任务清单
# ============================================================

# 1. 训练一棵无限制决策树和一棵 max_depth=4 的决策树
#    对比它们的训练/测试准确率
#    用 plot_tree() 可视化剪枝后的决策树
tree_unlimited=DecisionTreeClassifier(max_depth=None,random_state=42)
tree_unlimited.fit(X_train,y_train)
tree_limited=DecisionTreeClassifier(max_depth=4,random_state=42,min_samples_leaf=10,min_samples_split=20)
tree_limited.fit(X_train,y_train)
for name,model in [("无限制随机树",tree_unlimited),("有限制随机树",tree_limited)]:
    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)
    print(f"{name:<20}{train_score:<10.4f}{test_score:<10.4f}{model.get_n_leaves()}")
print("=" * 55)

fig, ax = plt.subplots(figsize=(20, 10))
# plot_tree(): 绘制决策树的图形化结构
# feature_names: 每个节点显示特征名而非特征编号
# class_names: 叶子节点显示类别名而非数字
# filled=True: 按预测类别填充颜色（纯度越高颜色越深）
# rounded=True: 使用圆角矩形
# proportion=True: 显示各类别的比例而非绝对数量
# fontsize=9: 字体大小
plot_tree(tree_limited,
          feature_names=X.columns,
          class_names=["遇难", "存活"],
          filled=True,
          rounded=True,
          proportion=True,
          fontsize=9,
          ax=ax)

ax.set_title("剪枝后的决策树（max_depth=4）", fontsize=16)
plt.tight_layout()
plt.show()
# 2. 训练随机森林，对比不同 n_estimators 的效果
#    n_estimators = [10, 50, 100, 200, 500]
#    绘制折线图（训���准确率 vs CV准确率）
train_result={}
test_result={}
model2={"10棵树":RandomForestClassifier(n_estimators=10,random_state=42),
        "50棵树":RandomForestClassifier(n_estimators=50,random_state=42),
        "100棵树":RandomForestClassifier(n_estimators=100,random_state=42),
        "200棵树":RandomForestClassifier(n_estimators=200,random_state=42),
        "500棵树":RandomForestClassifier(n_estimators=500,random_state=42)}
for name,model in model2.items():
    model.fit(X_train, y_train)
    train_score2=model.score(X_train,y_train)
    test_score2=model.score(X_test,y_test)
    train_result[name]=train_score2
    test_result[name]=test_score2
names=list(train_result.keys())
value1=list(train_result[name] for name in names)
value2=list(test_result[name] for name in names)
plt.plot(names,value1,marker="o",label="train")
plt.plot(names,value2,marker="s",label="test")
plt.xlabel('名称')
plt.ylabel('数值')
plt.title('两个字典的对比折线图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()
# 3. 同时比较决策树和随机森林的特征重要性
#    绘制两个模型的特征重要性条形图（2个子图）
#    观察：两者排名一致吗？随机森林的重要性分布更均匀吗？
rf_best = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_best.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 决策树特征重要性
tree_importances = pd.Series(tree_limited.feature_importances_, index=X.columns)
tree_importances.sort_values(ascending=True).plot(kind='barh', ax=axes[0], color='skyblue')
axes[0].set_title('决策树特征重要性 (max_depth=4)')
axes[0].set_xlabel('重要性')

# 随机森林特征重要性
rf_importances = pd.Series(rf_best.feature_importances_, index=X.columns)
rf_importances.sort_values(ascending=True).plot(kind='barh', ax=axes[1], color='lightcoral')
axes[1].set_title('随机森林特征重要性 (n_estimators=100)')
axes[1].set_xlabel('重要性')

plt.tight_layout()
plt.show()
# 4. 用最优随机森林在测试集上评估
#    打印 classification_report
#    绘制混淆矩阵

# 5. 回答问题（写在注释中）：
#    (a) 无限制决策树的训练准确率是多少？这正常吗？为什么？
#无限制决策树的训练正确率有98.46%，过高过拟合
#    (b) 随机森林中 n_estimators 从100增加到500，提升大吗？
#提升不大
#    (c) 决策树和随机森林哪个更适合做最终的生产模型？为什么？