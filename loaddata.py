import pandas as pd

# Load the dataset
diabetes_data = pd.read_csv("Diabetes.csv")

# Preview the first few rows
print(diabetes_data.head())
# Replace '?' with NaN
diabetes_data.replace("?", pd.NA, inplace=True)

# Convert bloodpressure to numeric
diabetes_data["bloodpressure"] = pd.to_numeric(diabetes_data["bloodpressure"], errors="coerce")

# Drop rows with missing values
diabetes_data_cleaned = diabetes_data.dropna()

# Preview cleaned data
print(diabetes_data_cleaned.info())

from sklearn.model_selection import train_test_split

# Features and target variable
X = diabetes_data_cleaned[["glucose", "bloodpressure"]]
y = diabetes_data_cleaned["diabetes"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

from sklearn.naive_bayes import GaussianNB

# Initialize the classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
