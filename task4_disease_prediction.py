# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load your dataset (replace 'your_data.csv' with the actual file name or path)
data = pd.read_csv('/Users/maria/Downloads/Covid_ Live.csv')
# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.shape[1])

# Separate features (X) and target variable (y)
X = data.drop('target_variable', axis=1)  # Replace 'target_variable' with the actual target column name
y = data['target_variable']

# Data Preprocessing
# Handle missing values and encode categorical variables if necessary
# You might need more advanced preprocessing depending on the characteristics of your data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Model Selection and Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_imputed, y_train)

# Model Evaluation
y_pred = model.predict(X_test_imputed)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)
