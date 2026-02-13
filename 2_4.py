import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# 加载数据
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ===== 子图1（左上）：各舱位的生存人数 =====
sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0,0])
axes[0,0].set_title('各舱位生存/遇难人数', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('舱位等级')
axes[0,0].set_ylabel('人数')
# 修改图例
handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,0].legend(handles, ['遇难', '存活'], title='状态')

# ===== 子图2（右上）：年龄分布直方图，按生存状态着色 =====
# 去除年龄为空的样本
df_age_clean = df.dropna(subset=['Age'])

# 在同一张图上画两个直方图
axes[1,0].hist(df_age_clean[df_age_clean['Survived']==1]['Age'], 
               bins=30, alpha=0.6, label='存活', color='green', edgecolor='black')
axes[1,0].hist(df_age_clean[df_age_clean['Survived']==0]['Age'], 
               bins=30, alpha=0.6, label='遇难', color='red', edgecolor='black')

axes[1,0].set_title('年龄分布：存活 vs 遇难', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('年龄')
axes[1,0].set_ylabel('人数')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# ===== 子图3（左下）：性别 vs 生存率 =====
# 计算性别生存率
gender_survival = df.groupby('Sex')['Survived'].mean().reset_index()
gender_survival['Survived'] = gender_survival['Survived'] * 100  # 转为百分比

# 画柱状图
sns.barplot(data=gender_survival, x='Sex', y='Survived', ax=axes[1,1], 
            palette=['#ff7f7f', '#7fbfff'])
axes[1,1].set_title('性别生存率对比', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('性别')
axes[1,1].set_ylabel('生存率 (%)')
# 在柱子上添加数值标签
for i, v in enumerate(gender_survival['Survived']):
    axes[1,1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=12)

# ===== 子图4（右下）：特征相关性热力图 =====
# 选择数值列
numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numeric_cols].corr()

# 画热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            ax=axes[0,1], square=True, cbar_kws={"shrink": 0.8})
axes[0,1].set_title('特征相关性热力图', fontsize=14, fontweight='bold')

# 调整布局（注意：子图索引需要重新排列）
# 因为上面我们用的是 axes[1,0] 和 axes[1,1] 作为左下和右下
# 但实际上子图排列是：
# axes[0,0] 左上 | axes[0,1] 右上
# axes[1,0] 左下 | axes[1,1] 右下
# 
# 所以我们把热力图放到 axes[0,1]（右上），
# 年龄分布放到 axes[1,0]（左下），
# 性别生存率放到 axes[1,1]（右下）

plt.suptitle("🚢 Titanic EDA 可视化报告", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=150, bbox_inches="tight")
plt.show()