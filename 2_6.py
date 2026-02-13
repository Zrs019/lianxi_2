from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分数据集（80%训练，20%测试，分层抽样，random_state=42）
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 创建至少3个不同的 Pipeline（每个包含 StandardScaler + 模型）
pipe1=Pipeline([("scaler",StandardScaler()),
                ("knn",KNeighborsClassifier(n_neighbors=5))])
pipe2=Pipeline([("scaler",StandardScaler()),
                ("Logi",LogisticRegression(max_iter=200))])
pipe3=Pipeline([("scaler",StandardScaler()),
                ("Dec",DecisionTreeClassifier(random_state=42))])
pipes=[pipe1,pipe2,pipe3]
pipe_names = ['KNN', 'Logistic Regression', 'Decision Tree']
#    例如：Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
# 4. 用 for 循环统一训练和评估，打印每个模型的准确率
print("=" * 40)
print(f"{'模型':<15} {'准确率':>8}")
print("=" * 40)
results = {}
for i,model in enumerate(pipes):
    model.fit(X_train, y_train) 
    accuracy=model.score(X_test,y_test)
    results[pipe_names[i]] = accuracy
    bar = "█" * int(accuracy * 30)
    print(f"{pipe_names[i]:<15} {accuracy:>8.4f}  {bar}")

print("=" * 40)
best = max(results, key=results.get)
print(f"🏆 最佳模型: {best} ({results[best]:.4f})")
# 5. 找出最佳模型，并用它对以下3朵"新花"做预测，输出预测的品种名称
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],   # 新花1
    [6.7, 3.1, 4.7, 1.5],   # 新花2
    [7.2, 3.6, 6.1, 2.5],   # 新花3
])
j=pipe_names.index(best)
best_pipe=pipes[j]
result_prediction=best_pipe.predict(new_flowers)
print(iris.target_names[result_prediction])
# 提示：iris.target_names[prediction] 可以把数字转为品种名