"""
任务2（动手版）：用 XGBoost 训练二分类模型，并通过“对照实验”掌握调参主线

你将完成三组对照：
A) 不同 learning_rate 对 best_iteration（最佳树数）与 AUC 的影响
B) 增大树复杂度（max_depth）是否更容易过拟合
C) 去掉采样随机性（subsample/colsample=1.0）是否更容易过拟合

你需要做的事：
1) 先直接运行，确保能看到输出表格
2) 然后按下面 TODO 提示改参数，观察 best_iteration 和 AUC 的变化
3) 最后在文件末尾的 TODO（注释区）写下你的结论（用注释回答即可）
"""

# ------------------------------
# 1. 导入依赖
# ------------------------------
import numpy as np  # 用于简单统计与格式化输出
from sklearn.datasets import load_breast_cancer  # sklearn 自带二分类数据集（方便跑通）
from sklearn.model_selection import train_test_split  # 划分训练集/验证集
from sklearn.metrics import roc_auc_score  # AUC 指标（你常用）
from xgboost import XGBClassifier  # XGBoost 的 sklearn API


# ------------------------------
# 2. 准备数据（训练/验证）
# ------------------------------
data = load_breast_cancer()
X = data.data
y = data.target

# early stopping 需要验证集；stratify=y 保持类别比例一致（很重要）
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------
# 3. 写一个“训练 + 返回指标”的函数（方便做对照实验）
# ------------------------------
def train_eval_xgb(
    learning_rate: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    early_stopping_rounds: int = 50,
    n_estimators: int = 5000,
    random_state: int = 42,
):
    """
    训练一个 XGBoost 模型，并返回：
    - best_iteration：早停找到的最佳迭代轮数（≈最佳树数）
    - valid_auc：验证集 AUC（越大越好）
    """

    # 构建模型：把关键参数都暴露出来，便于你系统对照
    model = XGBClassifier(
        objective="binary:logistic",   # 二分类
        eval_metric="auc",             # 训练过程监控 AUC

        learning_rate=learning_rate,   # 步长
        n_estimators=n_estimators,     # 先设很大，交给 early stopping 决定停在哪

        max_depth=max_depth,           # 树复杂度：越大越容易过拟合
        min_child_weight=min_child_weight,  # 越大越保守

        subsample=subsample,           # 样本采样比例：<1 引入随机性，缓解过拟合
        colsample_bytree=colsample_bytree,  # 特征采样比例：<1 引入随机性，缓解过拟合

        reg_lambda=reg_lambda,         # L2 正则：越大越保守

        n_jobs=-1,                     # 用满 CPU
        random_state=random_state,
        
        early_stopping_rounds=early_stopping_rounds,
    )

    # 训练：传入验证集 + early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],  # 训练过程中每轮都会在 valid 上算一次 AUC
        verbose=False                   # 设 True 可看每轮日志；这里先安静输出结果表
    )

    # 用最佳轮数的模型在验证集算 AUC
    valid_proba = model.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, valid_proba)

    return model.best_iteration, valid_auc


# ------------------------------
# 4. 对照实验 A：learning_rate 变化 -> best_iteration 与 AUC 如何变化？
# ------------------------------
# 固定树复杂度与采样策略，只改 learning_rate
settings_A = [
    {"learning_rate": 0.2, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.1, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.03, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
]

print("\n=== Experiment A: learning_rate trade-off ===")
print("lr    | max_depth | subsample | colsample | best_iter | valid_auc")
for s in settings_A:
    best_iter, auc = train_eval_xgb(**s)
    print(f"{s['learning_rate']:<5} | {s['max_depth']:<9} | {s['subsample']:<9} | {s['colsample_bytree']:<9} | {best_iter:<9} | {auc:.5f}")


# ------------------------------
# 5. 对照实验 B：树变复杂（max_depth 变大）会怎样？
# ------------------------------
# 固定 lr 与采样策略，只改 max_depth
settings_B = [
    {"learning_rate": 0.05, "max_depth": 2, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.05, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
]

print("\n=== Experiment B: tree complexity (max_depth) ===")
print("lr    | max_depth | subsample | colsample | best_iter | valid_auc")
for s in settings_B:
    best_iter, auc = train_eval_xgb(**s)
    print(f"{s['learning_rate']:<5} | {s['max_depth']:<9} | {s['subsample']:<9} | {s['colsample_bytree']:<9} | {best_iter:<9} | {auc:.5f}")


# ------------------------------
# 6. 对照实验 C：去掉采样随机性（subsample/colsample=1.0）会怎样？
# ------------------------------
# 固定 lr 与 max_depth，对比有无采样
settings_C = [
    {"learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8},
    {"learning_rate": 0.05, "max_depth": 4, "subsample": 1.0, "colsample_bytree": 1.0},
]

print("\n=== Experiment C: sampling randomness ===")
print("lr    | max_depth | subsample | colsample | best_iter | valid_auc")
for s in settings_C:
    best_iter, auc = train_eval_xgb(**s)
    print(f"{s['learning_rate']:<5} | {s['max_depth']:<9} | {s['subsample']:<9} | {s['colsample_bytree']:<9} | {best_iter:<9} | {auc:.5f}")


# ------------------------------
# 7. 你要做的“动手修改”（只改这里，别的先不动）
# ------------------------------
"""
TODO(动手1)：把 early_stopping_rounds 从 50 改成 10，再跑一遍：
- 观察 best_iter 是否更小？AUC 是否可能更差？
- 理解：早停太“急”可能停早了；太“松”会多训练一些但不一定更差（取决于数据/噪声）

TODO(动手2)：把 settings_A 里的 learning_rate 再加一个 0.01：
- 观察 best_iter 是否明显增大？
- 如果 best_iter 很大但 AUC 没提升，说明什么？

TODO(动手3)：把 settings_B 的 max_depth=6 再改成 8：
- 观察 AUC 是否不升反降（过拟合更明显的信号之一）
"""


# ------------------------------
# 8. 注释作答区（你要求：把问答放在代码最后，用注释回答）
# ------------------------------
"""
TODO(回答，用注释写在这里即可)：

Q1：在你的输出中，learning_rate 变小后 best_iter 一般会怎么变化？为什么？
A1：lr变小后best_iter增大，因为每棵树的贡献越小就需要更多树

Q2：你观察到 max_depth 变大时，valid_auc 一定会变好吗？如果没有，最可能原因是什么？
A2：会

Q3：subsample/colsample 从 0.8 提到 1.0 后，你看到的变化是什么？这说明采样随机性在起什么作用？
A3：总树数增加，但valid_auc反而降低，说明出现了过拟合，随机采样可以防止过拟合

Q4：请写出你自己的一条“实战调参顺序”（不超过 4 步），要求包含 early stopping。
A4：1.先保持树深度不变，保持随机采样，找到合适的lr。2.在lr保持不变情况下，尝试不同树深度，找最佳深度。3.确定lr和树深度后测试随机采样是否要开启。最后加入early stopping防止过拟合
"""