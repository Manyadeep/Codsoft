import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv(r"C:\intern3\Iris_dataset.csv")

print("Shape of dataset:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# Encode target labels: setosa → 0, versicolor → 1, virginica → 2
df['species'] = df['species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
print("\nEncoded target column:\n", df['species'].value_counts())

print("\nDataset Head:")
print(df.head())

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\nCross-Validation Accuracy (mean):", cv_scores.mean())
