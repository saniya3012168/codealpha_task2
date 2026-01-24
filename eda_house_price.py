import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Dataset
# -------------------------------
# Make sure Housing.csv is in the same folder as this file
df = pd.read_csv("Housing.xlsx")

# -------------------------------
# Basic Data Exploration
# -------------------------------
print("First 5 Rows:\n", df.head())
print("\nDataset Description:\n", df.describe())
print("\nDataset Info:")
df.info()
print("\nShape of Dataset:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# Price Distribution
# -------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Price vs Area
# -------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(x='area', y='price', data=df)
plt.title("Price vs Area")
plt.show()

# -------------------------------
# Price vs Air Conditioning
# -------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(x='airconditioning', y='price', data=df)
plt.title("Price vs Air Conditioning")
plt.show()

# -------------------------------
# Encoding Categorical Variables
# -------------------------------
df_encoded = df.copy()

varlist = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

def binary_map(x):
    return x.map({'yes': 1, 'no': 0})

df_encoded[varlist] = df_encoded[varlist].apply(binary_map)

df_encoded = pd.get_dummies(
    df_encoded,
    columns=['furnishingstatus'],
    drop_first=True
)

# -------------------------------
# Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(
    df_encoded.corr(numeric_only=True),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Furnishing Status Comparison
# -------------------------------
furnished = df[df['furnishingstatus'] == 'furnished']['price']
unfurnished = df[df['furnishingstatus'] == 'unfurnished']['price']

print("\nAverage Price (Furnished):", furnished.mean())
print("Average Price (Unfurnished):", unfurnished.mean())

# -------------------------------
# Outlier Detection
# -------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['price'])
plt.title("Price Outliers")
plt.show()
