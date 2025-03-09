import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# 데이터 로드
file_path = "C:\data\iris.csv"
df = pd.read_csv(file_path)



# 'Name' 열이 종속 변수(y), 나머지 열이 독립 변수(X)로 설정.
X = df.drop("Name", axis=1)  
y = df["Name"]      
# 데이터 분할(훈련 80%)     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. 의사결정나무 (Decision Tree)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("==================== Decision Tree ========================")
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.2f}")
print(classification_report(y_test, dt_pred),"\n\n")

# 2. 랜덤 포레스트 (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("==================== Random Forest ========================")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(classification_report(y_test, rf_pred),"\n\n")

# 3. 서포트 벡터 머신 (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print("================ Support Vector Machine ====================")
print(f"Accuracy: {accuracy_score(y_test, svm_pred):.2f}")
print(classification_report(y_test, svm_pred),"\n\n")

# 4. 로지스틱 회귀 (Logistic Regression)
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("================= Logistic Regression ====================")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.2f}")
print(classification_report(y_test, lr_pred))
