import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# تحميل البيانات
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')

# ترميز الأعمدة النصية
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

X = df.drop('y', axis=1)
y = df['y']

# عدد التكرارات
runs = 10

# لتخزين النتائج
train_acc_log, val_acc_log, test_acc_log = [], [], []
train_acc_perc, val_acc_perc, test_acc_perc = [], [], []
train_acc_mlp, val_acc_mlp, test_acc_mlp = [], [], []
best_models = []

for i in range(runs):
    # خلط البيانات عشوائياً في كل جولة
    X_shuffled, y_shuffled = shuffle(X, y)

    # تقسيم عشوائي بدون random_state
    X_temp, X_test, y_temp, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000, solver='liblinear')
    log_model.fit(X_train, y_train)
    train_log = log_model.score(X_train, y_train)
    val_log = log_model.score(X_val, y_val)
    test_log = log_model.score(X_test, y_test)
    train_acc_log.append(train_log)
    val_acc_log.append(val_log)
    test_acc_log.append(test_log)

    # Perceptron
    perc_model = Perceptron(max_iter=1000)
    perc_model.fit(X_train, y_train)
    train_perc = perc_model.score(X_train, y_train)
    val_perc = perc_model.score(X_val, y_val)
    test_perc = perc_model.score(X_test, y_test)
    train_acc_perc.append(train_perc)
    val_acc_perc.append(val_perc)
    test_acc_perc.append(test_perc)

    # MLPClassifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    mlp_model.fit(X_train, y_train)
    train_mlp = mlp_model.score(X_train, y_train)
    val_mlp = mlp_model.score(X_val, y_val)
    test_mlp = mlp_model.score(X_test, y_test)
    train_acc_mlp.append(train_mlp)
    val_acc_mlp.append(val_mlp)
    test_acc_mlp.append(test_mlp)

    # اختيار النموذج الأفضل بناءً على validation accuracy
    val_scores = {
        'Logistic Regression': val_log,
        'Perceptron': val_perc,
        'MLPClassifier': val_mlp
    }
    best_model = max(val_scores, key=val_scores.get)
    best_models.append(best_model)

    # طباعة دقة كل جولة
    print(f"Run {i+1}")
    print(f"  Logistic     => Train: {train_log:.4f}, Val: {val_log:.4f}, Test: {test_log:.4f}")
    print(f"  Perceptron   => Train: {train_perc:.4f}, Val: {val_perc:.4f}, Test: {test_perc:.4f}")
    print(f"  MLPClassifier=> Train: {train_mlp:.4f}, Val: {val_mlp:.4f}, Test: {test_mlp:.4f}")
    print()

# حساب المتوسطات والانحراف المعياري
def report(name, train, val, test):
    print(f"{name} - Train Avg: {np.mean(train):.4f}, Std: {np.std(train):.4f}")
    print(f"{name} - Val   Avg: {np.mean(val):.4f}, Std: {np.std(val):.4f}")
    print(f"{name} - Test  Avg: {np.mean(test):.4f}, Std: {np.std(test):.4f}")
    print()

print("📊 Average Accuracies and Standard Deviations:\n")
report("Logistic Regression", train_acc_log, val_acc_log, test_acc_log)
report("Perceptron", train_acc_perc, val_acc_perc, test_acc_perc)
report("MLPClassifier", train_acc_mlp, val_acc_mlp, test_acc_mlp)

# رسم مخطط الأعمدة
models = ['Logistic Regression', 'Perceptron', 'MLPClassifier']
train_accuracies = [np.mean(train_acc_log), np.mean(train_acc_perc), np.mean(train_acc_mlp)]
val_accuracies = [np.mean(val_acc_log), np.mean(val_acc_perc), np.mean(val_acc_mlp)]
test_accuracies = [np.mean(test_acc_log), np.mean(test_acc_perc), np.mean(test_acc_mlp)]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, train_accuracies, width, label='Train Accuracy', color='#99ccff')
bars2 = ax.bar(x, val_accuracies, width, label='Validation Accuracy', color='#66cc99')
bars3 = ax.bar(x + width, test_accuracies, width, label='Test Accuracy', color='#ffcc99')

ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Train, Validation, and Test Accuracy by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

for bar in bars1 + bars2 + bars3:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
