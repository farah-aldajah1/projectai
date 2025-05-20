
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# تحميل البيانات
df = pd.read_csv(r"C:\Users\LENOVO\OneDrive - Balqa Applied University\Desktop\Bank-full.csv", sep=';')

# تحويل البيانات النصية إلى أرقام باستخدام LabelEncoder
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# تقسيم البيانات إلى ميزات X والهدف y
X = df.drop('y', axis=1)
y = df['y']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# إنشاء نموذج Decision Tree وتدريبه
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# التنبؤ على مجموعة الاختبار
y_pred = model.predict(X_test)

# عرض الدقة والتقرير
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# رسم المصفوفة
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
