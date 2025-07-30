import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r"C:\intern\Titanic-Dataset.csv")

print("Shape of dataset:", df.shape)
print("\nColumn Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Encode 'Sex': male → 1, female → 0
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# Encode 'Embarked': C → 0, Q → 1, S → 2
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print(df.head())


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 1  
df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

# X = all columns except 'Survived'
X = df.drop('Survived', axis=1)
# y = target variable
y = df['Survived']
 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split( X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Print Accuracy
print(" Accuracy:", accuracy_score(y_test, y_pred))

# Print Classification Report
print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Print Confusion Matrix
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("\n Cross-Validation Accuracy (mean):", cv_scores.mean())

