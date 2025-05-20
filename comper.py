
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# تحميل البيانات
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')


# تحويل البيانات النصية إلى أرقام
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# فصل الميزات والهدف
X = df.drop('y', axis=1)
y = df['y']

# عدد التكرارات
runs = 10

# لتخزين الدقة لكل نموذج
acc_logistic = []
acc_perceptron = []
acc_mlp = []

for i in range(runs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, solver='liblinear')
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    acc_logistic.append(accuracy_score(y_test, y_pred_log))

    # Perceptron
    perc_model = Perceptron(max_iter=1000, random_state=i)
    perc_model.fit(X_train, y_train)
    y_pred_perc = perc_model.predict(X_test)
    acc_perceptron.append(accuracy_score(y_test, y_pred_perc))

    # MLPClassifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=i)
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    acc_mlp.append(accuracy_score(y_test, y_pred_mlp))

# حساب المتوسط
avg_log = np.mean(acc_logistic)
avg_perc = np.mean(acc_perceptron)
avg_mlp = np.mean(acc_mlp)

# طباعة النتائج
print(f"Average Logistic Regression Accuracy: {avg_log:.2f}")
print(f"Average Perceptron Accuracy: {avg_perc:.2f}")
print(f"Average MLP Accuracy: {avg_mlp:.2f}")

# رسم المقارنة
plt.figure(figsize=(12, 6))
plt.plot(range(1, runs+1), acc_logistic, marker='o', label='Logistic Regression')
plt.plot(range(1, runs+1), acc_perceptron, marker='s', label='Perceptron')
plt.plot(range(1, runs+1), acc_mlp, marker='^', label='MLPClassifier')
plt.axhline(avg_log, color='blue', linestyle='--', label=f'Avg Logistic = {avg_log:.2f}')
plt.axhline(avg_perc, color='orange', linestyle='--', label=f'Avg Perceptron = {avg_perc:.2f}')
plt.axhline(avg_mlp, color='green', linestyle='--', label=f'Avg MLP = {avg_mlp:.2f}')
plt.title('Model Accuracy Comparison over 10 Runs')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.xticks(range(1, runs+1))
plt.legend()
plt.grid(True)
plt.show()
