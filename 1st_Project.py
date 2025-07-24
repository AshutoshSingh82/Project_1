import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset (you can download from Kaggle or use seaborn if needed)
# For now, assume the file is "titanic.csv"
data = pd.read_csv("titanic.csv")

# Show first 5 rows
print(data.head())



# Select useful columns
data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Convert 'Sex' to number: male = 0, female = 1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Fill missing age with average age
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Drop any rows still having missing values
data.dropna(inplace=True)




# Separate input (X) and output (y)
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Random Forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%")



