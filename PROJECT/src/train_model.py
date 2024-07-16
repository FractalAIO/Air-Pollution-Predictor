import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from data_processing import load_dataset, clean_dataset, feature_engineer
from model import save_model

# Load the dataset
dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
print("Loaded dataset columns:", dataset.columns)

# Clean the dataset
cleaned_dataset = clean_dataset(dataset)
print("Cleaned dataset columns:", cleaned_dataset.columns)

# Perform feature engineering
engineered_dataset = feature_engineer(cleaned_dataset)
print("Engineered dataset columns:", engineered_dataset.columns)

# Ensure 'PM2.5 AQI Value' is in the dataset
if 'PM2.5 AQI Value' not in engineered_dataset.columns:
    raise ValueError("'PM2.5 AQI Value' is not in the dataset")

# Identify categorical columns
categorical_cols = ['AQI Category', 'CO AQI Category', 'NO2 AQI Category', 'Ozone AQI Category', 'PM2.5 AQI Category']

# Identify numeric columns
numeric_cols = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
print("Numeric columns:", numeric_cols)

# Verify all columns are present before proceeding
for col in categorical_cols + numeric_cols:
    if col not in engineered_dataset.columns:
        raise ValueError(f"Column '{col}' is not present in the dataset")

# Separate features and target
X = engineered_dataset.drop(columns=['PM2.5 AQI Value'])
y = engineered_dataset['PM2.5 AQI Value']
print("Features (X) columns before preprocessing:", X.columns)
print("Target (y) column:", y.name)

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# Create a pipeline with the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

# Verify that the feature columns are correct before fitting the pipeline
expected_features = set(categorical_cols + numeric_cols) - {'PM2.5 AQI Value'}
actual_features = set(X.columns)
if not expected_features.issubset(actual_features):
    missing_features = expected_features - actual_features
    raise ValueError(f"The following expected features are missing: {missing_features}")

# Fit the pipeline
pipeline.fit(X, y)

# Save the pipeline
save_model(pipeline, 'C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\model_pipeline.joblib')

print("Model trained and saved as model_pipeline.joblib")
