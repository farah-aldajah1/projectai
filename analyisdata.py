import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# عرض بيانات عامة عن الملف
print(df.head() )         # أول 5 صفوف
print(df.info())          # معلومات عن الأعمدة والأنواع
print(df.isnull().sum() ) # عدد القيم الفارغة
print(df.describe())      # إحصائيات عامة

# رسم توزيع الأعمار
plt.hist(df['age'], bins=20, edgecolor='black')
plt.title('Distribution of the customer age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
#رسم توزيع العمل  
plt.figure(figsize=(12,6))
sns.countplot(x='job' , data=df)
plt.title('count of customer by job')
plt.xlabel('job')
plt.ylabel('count')
plt.xticks(rotation=45)
plt.show()
#رسم تحقق من الرصيد 
plt.figure(figsize=(10,6))
sns.boxplot(x='y', y='balance', data=df)
plt.title('Boxplot of balance by deposit outcome')
plt.xlabel('deposit outcome')
plt.ylabel('Balance')
plt.show()