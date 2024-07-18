import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from data_processing import load_dataset, clean_dataset, feature_engineer

# Paths to dataset and model
DATASET_PATH = 'AirPollution/PROJECT/global air pollution dataset.csv'
MODEL_PATH = 'AirPollution/PROJECT/model_pipeline.joblib'

# Load the dataset
dataset = load_dataset(DATASET_PATH)
cleaned_dataset = clean_dataset(dataset)
engineered_dataset = feature_engineer(cleaned_dataset)

# Print dataset columns to debug
print("Loaded dataset columns:", dataset.columns)
print("Cleaned dataset columns:", cleaned_dataset.columns)
print("Engineered dataset columns:", engineered_dataset.columns)

# Define the target and features
target = 'PM2.5 AQI Value'
features = engineered_dataset.drop(columns=[target]).columns

# Split the data
X = engineered_dataset[features]
y = engineered_dataset[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(exclude=['number']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
