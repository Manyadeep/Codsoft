import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r"C:\intern2\Movie_dataset.csv", encoding='latin1')

print("Shape of dataset:", df.shape)
print("\nColumn Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Clean and preprocess
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(float)

df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
df['Actor 2'] = le.fit_transform(df['Actor 2'])
df['Actor 3'] = le.fit_transform(df['Actor 3'])

# Clean the 'Year' column: remove brackets or symbols
df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop rows again if any nulls got introduced
df.dropna(inplace=True)


# Define features and target
X = df[['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print(" MAE (Mean Absolute Error):", round(mae, 2))
print(" RMSE (Root Mean Squared Error):", round(rmse, 2))
print(" R² Score:", round(r2, 2))

# Cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print("\n Cross-Validation R² (mean):", round(cv_scores.mean(), 2))



