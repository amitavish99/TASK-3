# TASK-3
CREDIT CARD FRAUD DETECTION
Certainly! Detecting fraudulent credit card transactions is a crucial application of machine learning, and it involves dealing with class imbalance and using classification algorithms. Below is a simplified guide using Python and scikit-learn:

1. Import Libraries
python
Copy code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
2. Load and Explore Data
python
Copy code
# Load your credit card fraud dataset
# Assuming you have a CSV file named 'credit_card_data.csv'
data = pd.read_csv('credit_card_data.csv')

# Display the first few rows of the dataset
print(data.head())
3. Data Preprocessing and Normalization
python
Copy code
# Handle missing values if any
data = data.dropna()

# Extract features and target variable
features = data.drop('Class', axis=1)
target = data['Class']

# Normalize numerical features
features = (features - features.mean()) / features.std()
4. Split the Dataset
python
Copy code
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
5. Handle Class Imbalance
python
Copy code
# Use RandomOverSampler for oversampling
# Use RandomUnderSampler for undersampling
# Here, we use a combination of oversampling and undersampling with a Random Forest classifier
model = Pipeline([
    ('oversampler', RandomOverSampler(sampling_strategy=0.5)),
    ('undersampler', RandomUnderSampler(sampling_strategy=0.5)),
    ('classifier', RandomForestClassifier(random_state=42))
])
6. Train the Model
python
Copy code
# Train the model
model.fit(X_train, y_train)
7. Evaluate the Model
python
Copy code
# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model performance
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
This example uses a Random Forest classifier and a combination of oversampling and undersampling to handle class imbalance. You can experiment with other classifiers and sampling techniques based on your dataset characteristics.
