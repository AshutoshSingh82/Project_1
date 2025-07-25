# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Step 2: Sample Data (You can replace this with a CSV file)
data = {
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Drama'],
    'Director': ['Director1', 'Director2', 'Director1', 'Director3', 'Director2'],
    'Actor': ['Actor1', 'Actor2', 'Actor3', 'Actor1', 'Actor2'],
    'Budget': [100, 30, 50, 120, 40],
    'Rating': [7.8, 6.5, 7.0, 8.2, 6.9]
}

df = pd.DataFrame(data)

# Step 3: Preprocessing - Label Encoding for Categorical Columns
le = LabelEncoder()
df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor'] = le.fit_transform(df['Actor'])

# Step 4: Features and Target
X = df[['Genre', 'Director', 'Actor', 'Budget']]
y = df['Rating']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 9: Predict New Movie
new_movie = pd.DataFrame({
    'Genre': le.transform(['Action']),
    'Director': le.transform(['Director1']),
    'Actor': le.transform(['Actor1']),
    'Budget': [90]
})
predicted_rating = model.predict(new_movie)
print("Predicted Rating:", predicted_rating[0])
