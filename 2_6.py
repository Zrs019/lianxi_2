# ============================================================
# 库导入（请补充需要的库）
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 任务清单
# ============================================================

# 1. 加载数据，打印前5行和基本统计信息（describe）
housing= fetch_california_housing()
X=housing.data
y=housing.target
df=pd.DataFrame(X,columns=housing.feature_names)
df["Price"]=y
print("查看前五行")
print(df.head())
print(df.describe())
# 2. 划分训练集/测试集（80/20），标准化
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
scaler_X_train=scaler.fit_transform(X_train)
scaler_X_test=scaler.transform(X_test)
# 3. 分别训练 LinearRegression、Ridge(alpha=1.0)、Lasso(alpha=0.01)
models={"线性回归":LinearRegression(),"Ridge(alpha=1.0)":Ridge(alpha=100),"Lasso(alpha=0.01)":Lasso(alpha=100)}
print("=" * 65)
print(f"{'模型':<20} {'训练R²':>8} {'测试R²':>8} {'测试RMSE':>10}")
print("=" * 65)
results={}
for name,model in models.items():
    model.fit(scaler_X_train,y_train)
    train_r2=model.score(scaler_X_train,y_train)
    test_r2=model.score(scaler_X_test,y_test)
    y_pred=model.predict(scaler_X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    results[name]={"train_r2":train_r2,"test_r2":test_r2,"rmse":rmse}
# 4. 打印三个模型的 训练R²、测试R²、测试RMSE，做成对比表格
    print(f"{name:<20} {train_r2:>8.4f} {test_r2:>8.4f} {rmse:>10.4f}")
# 5. 绘制三个模型的权重柱状图（3个子图）
#    观察：Lasso 是否有权重被压成了 0？哪些特征最重要？
print("=" * 65)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
feature_names=housing.feature_names
# 6. 选出表现最好的模型，绘制「预测值 vs 真实值」散点图
for idx,(name,model) in enumerate(
    [("线性回归",models["线性回归"]),
    ("Ridge(alpha=1.0)",models["Ridge(alpha=1.0)"]),
    ("Lasso(alpha=0.01)",models["Lasso(alpha=0.01)"])]             
    ):
    coefs=model.coef_
    sorted_idx = np.argsort(np.abs(coefs))   # 按绝对值排序
    colors = ["#f44336" if c < 0 else "#4CAF50" for c in coefs[sorted_idx]]

    # barh(): 水平柱状图
    axes[idx].barh(range(len(coefs)), coefs[sorted_idx], color=colors)
    axes[idx].set_yticks(range(len(coefs)))
    axes[idx].set_yticklabels(np.array(feature_names)[sorted_idx])
    axes[idx].set_title(f"{name}\n权重分布", fontsize=13)
    axes[idx].axvline(x=0, color="black", linewidth=0.5)   # 画 x=0 竖线

    # 标注被 Lasso 压到0的特征
    if "Lasso" in name:
        zero_count = np.sum(np.abs(coefs) < 1e-10)
        axes[idx].set_xlabel(f"（{zero_count}个特征被压为0）")

plt.suptitle("三种回归模型的权重对比", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()
# 7. 回答问题（用注释写在代码中）：
#    (a) 三个模型表现差距大吗？为什么？
    #差距不大
#    (b) 哪些特征对房价影响最大？（看权重绝对值）
    #Latitude,Longitude,medlnk对房价影响最大
#    (c) 如果 alpha 设成 100，模型会怎样？（试一下）
#   改成100后，Lasso模型所有特征的权重都变成了0