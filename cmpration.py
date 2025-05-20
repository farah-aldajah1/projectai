
import matplotlib.pyplot as plt
import numpy as np

# هذه الدقة هي نتيجة النماذج بعد 10 تجارب (يتم حسابها من الكود السابق)
avg_log = np.mean(acc_logistic)
avg_perc = np.mean(acc_perceptron)
avg_mlp = np.mean(acc_mlp)

# أسماء النماذج
models = ['Logistic Regression', 'Perceptron', 'MLPClassifier']
# متوسط الدقة لكل نموذج
accuracies = [avg_log, avg_perc, avg_mlp]

# رسم الأعمدة
plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['skyblue', 'orange', 'lightgreen'])

# إضافة قيم الدقة فوق الأعمدة
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.2f}', ha='center', fontsize=12)

plt.title('Average Accuracy of Models (Bar Chart)', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
