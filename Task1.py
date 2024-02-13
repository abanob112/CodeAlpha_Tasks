# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your historical financial data into a Pandas DataFrame
# Assume the data has features (X) and labels (y)
# Replace 'your_data.csv' with the actual file path or URL
data = pd.read_csv('your_data.csv')

# Split the data into training and testing sets
X = data.drop('creditworthiness_label', axis=1)  # Assuming 'creditworthiness_label' is the target variable
y = data['creditworthiness_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier model (you can try other classifiers as well)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display classification report for more detailed metrics
print('Classification Report:\n', classification_report(y_test, y_pred))

