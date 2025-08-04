from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

print(df.head())

print(df.info())
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'].fillna(df['Embarked'].mode()[0])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'])

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df = remove_outliers_iqr(df, 'Age')
df = remove_outliers_iqr(df, 'Fare')

sns.set(style="whitegrid")
plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.xticks([0, 1], ['Male', 'Female'])

plt.subplot(2, 3, 2)
survived_counts = df['Survived'].value_counts()
plt.pie(survived_counts, labels=['Died', 'Survived'], autopct='%1.1f%%', startangle=90)
plt.title("Overall Survival Rate")

plt.tight_layout()
plt.suptitle("Titanic Dataset - Combined EDA Plots", fontsize=16, y=1.02)
plt.show()
