# eda_titanic.ipynb or main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# 2. Summary Statistics
print("Basic Info:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe(include='all'))

# 3. Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Data Cleaning (optional)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 5. Univariate Plots
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(data=df, x='Pclass', y='Age')
plt.title("Boxplot of Age by Pclass")
plt.show()

# 6. Categorical Feature Distribution
sns.countplot(data=df, x='Survived')
plt.title("Survival Count")
plt.show()

# 7. Multivariate Analysis
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']])
plt.show()

# 8. Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 9. Detecting Skewness
from scipy.stats import skew
print("\nSkewness:")
for col in ['Age', 'Fare']:
    print(f"{col}: {skew(df[col].dropna())}")

# 10. Plotly interactive plot
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title="Interactive Scatter Plot")
fig.show()
