import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (1) countplot：分类计数
sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 0],
              palette={0: "#f44336", 1: "#4CAF50"})
axes[0, 0].set_title("各舱位生存人数")
axes[0, 0].set_xticklabels(["一等舱", "二等舱", "三等舱"])

# (2) boxplot：箱线图（看分布 + 异常值）
sns.boxplot(data=df, x="Pclass", y="Age", hue="Survived", ax=axes[0, 1],
            palette={0: "#f44336", 1: "#4CAF50"})
axes[0, 1].set_title("各舱位年龄分布（按生存）")

# (3) violinplot：小提琴图（箱线图 + 密度图）
sns.violinplot(data=df, x="Sex", y="Age", hue="Survived", split=True,
               ax=axes[1, 0], palette={0: "#f44336", 1: "#4CAF50"})
axes[1, 0].set_title("性别年龄分布（按生存）")

# (4) kdeplot：核密度估计图
sns.kdeplot(data=df[df["Survived"]==1], x="Age", label="存活", 
            fill=True, alpha=0.4, color="#4CAF50", ax=axes[1, 1])
sns.kdeplot(data=df[df["Survived"]==0], x="Age", label="遇难",
            fill=True, alpha=0.4, color="#f44336", ax=axes[1, 1])
axes[1, 1].set_title("年龄密度分布（按生存）")
axes[1, 1].legend()

plt.suptitle("Titanic 数据探索性分析", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()