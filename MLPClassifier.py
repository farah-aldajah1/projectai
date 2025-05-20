
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# تحميل البيانات
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')

# ترميز الأعمدة النصية
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# فصل البيانات
X = df.drop('y', axis=1)
y = df['y']

# موازنة الخصائص
scaler = StandardScaler()
X = scaler.fit_transform(X)

# لتخزين النتائج
train_accuracies = []
test_accuracies = []

# تكرار 10 مرات
for i in range(10):
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    # إنشاء وتدريب النموذج
    mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=500, random_state=i)
    mlp.fit(X_train, y_train)

    # حساب الدقة
    train_acc = accuracy_score(y_train, mlp.predict(X_train))
    test_acc = accuracy_score(y_test, mlp.predict(X_test))

    # حفظ النتائج
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Run {i+1}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# متوسط الدقة
print("\nAverage Train Accuracy over 10 runs:", np.mean(train_accuracies))
print("Average Test Accuracy over 10 runs:", np.mean(test_accuracies))
import matplotlib.pyplot as plt

# رسم النتائج
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, 11), test_accuracies, marker='s', label='Test Accuracy')
plt.title('Train vs Test Accuracy over 10 Runs (MLPClassifier)')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.show()

