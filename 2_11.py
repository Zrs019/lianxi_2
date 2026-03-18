"""
任务3：Titanic 风格（数值+类别）特征工程 + XGBoost + Early Stopping + 特征重要性(gain)

你将学会：
1) 用 ColumnTransformer 同时处理数值/类别特征（可复用、避免泄漏）
2) 用 early stopping 自动找到最佳迭代轮数
3) 导出 one-hot 后的特征名，并结合 XGBoost 的 feature_importances_ 做 Top 特征分析

使用方式：
- 你可以先用自己的 Titanic DataFrame 替换示例部分
- 只要你提供：df（包含特征列）和 y（标签列 Survived），这套就能直接跑
"""

# ------------------------------
# 1. 导入依赖
# ------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier


# ------------------------------
# 2. 【示例】构造一个“Titanic 风格”的 DataFrame（你会替换为真实 Titanic）
# ------------------------------
# 注意：这里用假数据只是为了让模板可运行
# 你实际做 Titanic 时，直接读取 kaggle 的 train.csv，并保留 Survived 作为 y
df = pd.DataFrame({
    "Pclass": np.random.choice([1, 2, 3], size=800),
    "Sex": np.random.choice(["male", "female"], size=800),
    "Age": np.random.normal(30, 14, size=800),
    "SibSp": np.random.randint(0, 4, size=800),
    "Parch": np.random.randint(0, 3, size=800),
    "Fare": np.abs(np.random.normal(30, 50, size=800)),
    "Embarked": np.random.choice(["S", "C", "Q", None], size=800),
})

# 构造一个“伪标签”（只是为了模板演示能跑）
# 真 Titanic：y = df["Survived"]
y = (df["Sex"].eq("female").astype(int) ^ (df["Pclass"] == 3).astype(int)).values

X = df.copy()


# ------------------------------
# 3. 划分训练/验证（early stopping 需要验证集）
# ------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------
# 4. 区分数值列/类别列（Titanic 就是这种混合类型）
# ------------------------------
numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Sex", "Embarked"]

# 数值处理：缺失值 -> 中位数；再标准化（树模型不强依赖标准化，但做了也无妨）
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 类别处理：缺失值 -> 众数；再 OneHot
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 把两套处理拼起来
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"  # 其他列丢弃，避免不小心把泄漏列带进去
)


# ------------------------------
# 5. 定义 XGBoost（参数以“稳”为主）
# ------------------------------
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",

    learning_rate=0.05,
    n_estimators=5000,     # 设大，交给 early stopping
    max_depth=4,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_lambda=1.0,

    n_jobs=-1,
    random_state=42,
    use_label_encoder=False
)

# 注意：early stopping 需要把 eval_set 传给模型的 fit
# 但 Pipeline.fit 不能直接把 eval_set 传给内部模型，解决办法是使用参数前缀：model__xxx
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb)
])


# ------------------------------
# 6. 训练（关键：把验证集也通过同一个 preprocess 处理后喂给 eval_set）
# ------------------------------
# 先分别 transform，得到 numpy 特征矩阵（这样就能传给 XGBClassifier.fit 的 eval_set）
X_train_processed = preprocess.fit_transform(X_train)
X_valid_processed = preprocess.transform(X_valid)

# 直接训练底层 xgb（更直观），并保留 preprocess 供后面做特征名映射
xgb.fit(
    X_train_processed, y_train,
    eval_set=[(X_valid_processed, y_valid)],
    early_stopping_rounds=50,
    verbose=False
)

print("Best iteration:", xgb.best_iteration)

# 验证集 AUC
valid_proba = xgb.predict_proba(X_valid_processed)[:, 1]
valid_auc = roc_auc_score(y_valid, valid_proba)
print("Valid AUC:", round(valid_auc, 5))


# ------------------------------
# 7. 导出 one-hot 后的特征名 + 输出 Top 特征重要性
# ------------------------------
# 从 ColumnTransformer 里取出 OneHotEncoder，并拿到扩展后的类别特征名
ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]

# 注意：不同 sklearn 版本 get_feature_names_out 的调用方式略有不同
cat_feature_names = ohe.get_feature_names_out(categorical_features)

# 数值特征在经过 numeric_transformer 后，名字仍对应原列（我们直接用原名）
all_feature_names = np.concatenate([numeric_features, cat_feature_names])

# XGBoost 的 feature_importances_ 默认是基于 gain/weight 的某种汇总（版本不同略有差异）
importances = xgb.feature_importances_

# 取 Top N
top_n = 15
idx = np.argsort(importances)[::-1][:top_n]

print(f"\nTop {top_n} feature importances:")
for rank, i in enumerate(idx, start=1):
    print(f"{rank:02d}. {all_feature_names[i]} = {importances[i]:.6f}")


# ------------------------------
# 8. 注释作答区（按你的要求：在代码最后，用注释回答）
# ------------------------------
"""
TODO(回答，用注释写在这里即可)：

Q1：在 Titanic 这种中小表格数据上，你预计 XGBoost 相比随机森林的优势可能是什么？（至少写 2 点）
A1：

Q2：如果你看到 best_iteration 非常小（比如 < 20），你会优先怀疑哪些参数/问题？（至少写 2 个）
A2：

Q3：你在 Top 特征里看到了哪些类别 one-hot 特征？它们在业务上可能意味着什么？
A3：
"""