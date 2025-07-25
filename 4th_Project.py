# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Sample Advertising Dataset
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}

df = pd.DataFrame(data)

# Step 3: Show Data
print("Dataset:\n", df)

# Step 4: Visualize Correlation
sns.pairplot(df, kind='reg')
plt.show()

# Step 5: Feature and Target
X = df[['TV', 'Radio', 'Newspaper']]  # Inputs
y = df['Sales']  # Output

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predict
y_pred = model.predict(X_test)

# Step 9: Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 10: Predict Future Sales
# Suppose: TV=100, Radio=30, Newspaper=50
new_data = pd.DataFrame({'TV': [100], 'Radio': [30], 'Newspaper': [50]})
future_sales = model.predict(new_data)
print("Predicted Sales:", future_sales[0])
