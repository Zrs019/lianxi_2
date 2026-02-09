import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
train = pd.read_csv("D:/minicondadaima/lianxi/train.csv");
test = pd.read_csv("D:/minicondadaima/lianxi/test.csv");
#删除Cabin列
train=train.drop("Cabin",axis=1);
test=test.drop("Cabin",axis=1);
#查看缺失值数量
train_Age_median=train["Age"].median();
test_Age_median=test["Age"].median();
test_Fare_median=test["Fare"].median();
train_Embarked_mode=train["Embarked"].mode()[0];
test_Embarked_mode=test["Embarked"].mode()[0];
#填充缺失值
train['Age'] = train['Age'].fillna(train_Age_median);
test['Age'] = test['Age'].fillna(test_Age_median);
train['Embarked']=train['Embarked'].fillna(train_Embarked_mode);
test['Embarked']=test['Embarked'].fillna(test_Embarked_mode);
test['Fare']=test['Fare'].fillna(test_Fare_median);
# 新增家庭规模特征
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
# 是否独自一人
train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
test['IsAlone'] = (test['FamilySize'] == 1).astype(int)
# 从 Name 提取称谓 Title
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 合并低频称谓
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 
               'Rev', 'Sir', 'Jonkheer', 'Dona']
train['Title'] = train['Title'].replace(rare_titles, 'Rare')
test['Title'] = test['Title'].replace(rare_titles, 'Rare')

# 统一同义称谓
train['Title'] = train['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
test['Title'] = test['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
#one-hot编码
train=pd.get_dummies(train,columns=["Sex","Embarked","Title"]);
test=pd.get_dummies(test,columns=["Sex","Embarked","Title"]);
#识别目标列
target_column = 'Survived';
y1=train[target_column];
#对齐
def align_dataframes(df1, df2, method='intersection', fill_value=0, sort_columns=False):
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    # 根据方法确定最终列集合
    if method == 'intersection':
        final_cols = list(cols1 & cols2)
    elif method == 'union':
        final_cols = list(cols1 | cols2)
    elif method == 'left':
        final_cols = list(cols1)
    elif method == 'right':
        final_cols = list(cols2)
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 可选：对列排序
    if sort_columns:
        final_cols.sort()
    
    # 对齐两个DataFrame
    df1_aligned = df1.reindex(columns=final_cols, fill_value=fill_value)
    df2_aligned = df2.reindex(columns=final_cols, fill_value=fill_value)
    
    return df1_aligned, df2_aligned
df1 = train
df2 = test

print("原始df1:", df1.columns.tolist())
print("原始df2:", df2.columns.tolist())

df1_a, df2_a = align_dataframes(df1, df2, method='intersection')
#删除name和ticket
df1_a=df1_a.drop("Name",axis=1);
df1_a=df1_a.drop("Ticket",axis=1);
df2_a=df2_a.drop("Name",axis=1);
df2_a=df2_a.drop("Ticket",axis=1);
#识别特征列
X1=df1_a;
#三分化集，训练集70%，验证集15%，测试集15%
feature_names = X1.columns.tolist();
# 标准划分：训练集70%，验证集15%，测试集15%
def split_three_way(X, y, train_size=0.7, val_size=0.15, test_size=0.15, 
                    random_state=42, stratify=True):
    """
    将数据划分为训练集、验证集、测试集
    
    参数:
    X: 特征数据
    y: 目标值
    train_size: 训练集比例
    val_size: 验证集比例
    test_size: 测试集比例
    random_state: 随机种子
    stratify: 是否分层抽样（保持类别比例）
    
    返回:
    六个数据集：X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 验证比例总和为1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "比例总和必须为1"
    
    # 第一步：先分出测试集
    if stratify:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state
        )
    
    # 第二步：从剩余数据中分出验证集
    # 计算验证集相对于训练+验证集的比例
    val_relative_size = val_size / (train_size + val_size)
    
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_relative_size,
            random_state=random_state,
            stratify=y_train_val
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_relative_size,
            random_state=random_state
        )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 执行划分
X_train, X_val, X_test, y_train, y_val, y_test = split_three_way(
    X1, y1, 
    train_size=0.7, 
    val_size=0.15, 
    test_size=0.15,
    random_state=42,
    stratify=True
)
# 保存划分结果到文件
def save_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test, 
                        feature_names, output_dir='./data_splits/'):
    """
    保存划分后的数据集到CSV文件
    """
    import os
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    
    # 保存验证集
    val_df = X_val.copy()
    val_df['target'] = y_val
    val_df.to_csv(f'{output_dir}/validation.csv', index=False)
    
    # 保存测试集
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    # 保存数据信息
    info = {
        'total_samples': len(X_train) + len(X_val) + len(X_test),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': feature_names,
        'target_name': 'target'
    }
    
    import json
    with open(f'{output_dir}/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到 {output_dir} 目录")
    print(f"训练集: train.csv ({len(X_train)} 样本)")
    print(f"验证集: validation.csv ({len(X_val)} 样本)")
    print(f"测试集: test.csv ({len(X_test)} 样本)")

# 执行保存
save_split_datasets(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# 参数网格
param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8, None],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("交叉验证最佳准确率:", grid_search.best_score_)

# 用最佳模型评估验证集/测试集
best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)
print("验证集准确率:", accuracy_score(y_val, y_val_pred))
print("验证集分类报告:")
print(classification_report(y_val, y_val_pred))

y_test_pred = best_model.predict(X_test)
print("测试集准确率:", accuracy_score(y_test, y_test_pred))

#print(y1.head());
#print(df1_a.head());
#print(df2_a.head()); 