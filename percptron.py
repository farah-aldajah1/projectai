
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# تحميل البيانات
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')

# تحويل البيانات النصية إلى رقمية
df_encoded = df.copy()
label_encoders = {}

for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# فصل البيانات إلى ميزات وهدف
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

train_accuracies = []
test_accuracies = []

# التدريب 10 مرات
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

    model = Perceptron(max_iter=1000, tol=1e-3, random_state=i)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"Run {i+1}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

# حساب المتوسط
avg_train = np.mean(train_accuracies)
avg_test = np.mean(test_accuracies)

print(f"\nAverage Train Accuracy over 10 runs: {avg_train:.4f}")
print(f"Average Test Accuracy over 10 runs: {avg_test:.4f}")

# رسم النتائج
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), train_accuracies, marker='o', label='Train Accuracy')
plt.plot(range(1, 11), test_accuracies, marker='s', label='Test Accuracy')
plt.title('Train vs Test Accuracy over 10 Runs (Perceptron)')
plt.xlabel('Run Number')
plt.ylabel('Accuracy')
plt.ylim(0.5, 1.0)
plt.xticks(range(1, 11))
plt.legend()
plt.grid(True)
plt.show()
