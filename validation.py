
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

# ترميز الأعمدة النصية
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# المدخلات والمخرجات
X = df.drop('y', axis=1)
y = df['y']

# عدد التكرارات
runs = 10

# لتخزين النتائج
val_acc_log, val_acc_perc, val_acc_mlp = [], [], []
test_acc_log, test_acc_perc, test_acc_mlp = [], [], []
best_models = []

for i in range(runs):
    # تقسيم إلى: 60% تدريب، 20% تحقق، 20% اختبار
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=i)  # 0.25*0.8 = 0.2

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, solver='liblinear')
    log_model.fit(X_train, y_train)
    val_log = log_model.score(X_val, y_val)
    test_log = log_model.score(X_test, y_test)

    # Perceptron
    perc_model = Perceptron(max_iter=1000, random_state=i)
    perc_model.fit(X_train, y_train)
    val_perc = perc_model.score(X_val, y_val)
    test_perc = perc_model.score(X_test, y_test)

    # MLP Classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=i)
    mlp_model.fit(X_train, y_train)
    val_mlp = mlp_model.score(X_val, y_val)
    test_mlp = mlp_model.score(X_test, y_test)

    # حفظ النتائج
    val_acc_log.append(val_log)
    val_acc_perc.append(val_perc)
    val_acc_mlp.append(val_mlp)

    test_acc_log.append(test_log)
    test_acc_perc.append(test_perc)
    test_acc_mlp.append(test_mlp)

    # اختيار الأفضل في هذه الجولة
    val_scores = {
        'Logistic Regression': val_log,
        'Perceptron': val_perc,
        'MLPClassifier': val_mlp
    }
    best_model = max(val_scores, key=val_scores.get)
    best_models.append(best_model)

# المتوسطات
avg_val_log = np.mean(val_acc_log)
avg_val_perc = np.mean(val_acc_perc)
avg_val_mlp = np.mean(val_acc_mlp)

avg_test_log = np.mean(test_acc_log)
avg_test_perc = np.mean(test_acc_perc)
avg_test_mlp = np.mean(test_acc_mlp)

# طباعة النتائج
print("Average Validation Accuracies:")
print(f"Logistic Regression: {avg_val_log:.2f}")
print(f"Perceptron: {avg_val_perc:.2f}")
print(f"MLPClassifier: {avg_val_mlp:.2f}\n")

print("Average Test Accuracies:")
print(f"Logistic Regression: {avg_test_log:.2f}")
print(f"Perceptron: {avg_test_perc:.2f}")
print(f"MLPClassifier: {avg_test_mlp:.2f}\n")

print("Best Model per run (based on validation set):")
print(best_models)

# رسم بياني للمقارنة
plt.figure(figsize=(12, 6))
plt.plot(range(1, runs+1), val_acc_log, marker='o', label='Validation - Logistic')
plt.plot(range(1, runs+1), val_acc_perc, marker='s', label='Validation - Perceptron')
plt.plot(range(1, runs+1), val_acc_mlp, marker='^', label='Validation - MLP')

plt.plot(range(1, runs+1), test_acc_log, marker='o', linestyle='--', label='Test - Logistic')
plt.plot(range(1, runs+1), test_acc_perc, marker='s', linestyle='--', label='Test - Perceptron')
plt.plot(range(1, runs+1), test_acc_mlp, marker='^', linestyle='--', label='Test - MLP')

plt.title('Validation vs Test Accuracy Comparison (10 Runs)')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.xticks(range(1, runs+1))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# بيانات الأعمدة
models = ['Logistic Regression', 'Perceptron', 'MLPClassifier']
val_accuracies = [avg_val_log, avg_val_perc, avg_val_mlp]
test_accuracies = [avg_test_log, avg_test_perc, avg_test_mlp]

x = np.arange(len(models))  # مواضع الأعمدة
width = 0.35  # عرض كل عمود

# إنشاء الرسم
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy', color='skyblue')
bars2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='lightgreen')

# إعدادات الشكل
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Average Validation and Test Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)  # لأن الدقة تكون بين 0 و 1
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# عرض القيم على الأعمدة
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

