# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  # for handling imbalance

# Step 2: Load Dataset (Use real CSV in practice)
# Simulated small dataset for demo
data = {
    'Amount': [100, 2000, 150, 3000, 80, 6000, 40, 1200],
    'Time': [5, 10, 15, 20, 25, 30, 35, 40],
    'Feature1': [0.1, 1.2, 0.3, 1.5, 0.05, 2.0, 0.02, 0.8],
    'Feature2': [0.5, 1.8, 0.6, 1.6, 0.1, 2.5, 0.03, 1.1],
    'Class': [0, 1, 0, 1, 0, 1, 0, 0]  # 1 = fraud, 0 = genuine
}

df = pd.DataFrame(data)

# Step 3: Normalize Features
scaler = StandardScaler()
features = ['Amount', 'Time', 'Feature1', 'Feature2']
df[features] = scaler.fit_transform(df[features])

# Step 4: Split X and y
X = df.drop('Class', axis=1)
y = df['Class']

# Step 5: Handle Class Imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Step 6: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predict and Evaluate
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
