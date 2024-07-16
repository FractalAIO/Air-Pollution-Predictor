import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# Model training function
def train_model(data, target_column):
    # Select only numeric columns for training
    X = data.select_dtypes(include=['number']).drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Example with Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Example with Random Forest Regressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model, X_test, y_test

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    
    return rmse, mae

# Function to save the model
def save_model(model, file_path):
    joblib.dump(model, file_path)

# Function to load the model
def load_model(file_path):
    return joblib.load(file_path)
