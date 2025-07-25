# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Step 3: Display Basic Info
print("First 5 rows of dataset:\n", df.head())

# Step 4: Data Visualization (optional)
sns.pairplot(df, hue='species')
plt.show()

# Step 5: Prepare Data
X = df.drop('species', axis=1)
y = df['species']

# Step 6: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 7: Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the Model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Predict New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]  # Sample measurements
prediction = model.predict(sample)
print("Predicted Species:", prediction[0])
