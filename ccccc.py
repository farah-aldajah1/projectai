
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# تحميل البيانات

df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')

# ترميز الأعمدة النصية
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop('y', axis=1)
y = df['y']

# =================== Logistic Regression ===================
acc_logistic = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)
    acc_logistic.append(accuracy_score(y_test, model.predict(X_test)))

# =================== Perceptron ===================
acc_perceptron = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = Perceptron(max_iter=1000)
    model.fit(X_train, y_train)
    acc_perceptron.append(accuracy_score(y_test, model.predict(X_test)))

# =================== MLPClassifier ===================
acc_mlp = []
for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    model.fit(X_train, y_train)
    acc_mlp.append(accuracy_score(y_test, model.predict(X_test)))

# =================== Plotting ===================
avg_log = np.mean(acc_logistic)
avg_perc = np.mean(acc_perceptron)
avg_mlp = np.mean(acc_mlp)

models = ['Logistic Regression', 'Perceptron', 'MLPClassifier']
accuracies = [avg_log, avg_perc, avg_mlp]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'orange', 'lightgreen'])

# إضافة القيم فوق الأعمدة
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', ha='center', fontsize=12)

plt.title('Average Accuracy of Models (Bar Chart)', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
