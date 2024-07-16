import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_dataset, clean_dataset, feature_engineer
from src.model import load_model

# Function to predict air pollution for the whole country
def predict_pollution(country):
    # Load the dataset
    dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
    cleaned_dataset = clean_dataset(dataset)
    engineered_dataset = feature_engineer(cleaned_dataset)
    
    # Print unique countries to debug
    print("Unique countries in dataset:", engineered_dataset['Country'].unique())
    
    # Check if the country exists in the dataset
    input_data = engineered_dataset[engineered_dataset['Country'].str.lower() == country.lower()]
    
    # Debugging statements to print filtered data
    print(f"Filtered data for {country}:")
    print(input_data)
    
    if input_data.empty:
        messagebox.showerror("Error", "No data available for the selected country.")
        return
    
    # Select only numeric columns for aggregation
    numeric_columns = input_data.select_dtypes(include=['number']).columns
    country_data = input_data.groupby('Country')[numeric_columns].mean().reset_index()
    
    # Ensure PM2.5 AQI Value is not included in the features for prediction
    X_country = country_data.drop(columns=['Country', 'PM2.5 AQI Value'])
    
    # Load the trained pipeline
    pipeline = load_model('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\model_pipeline.joblib')
    
    # Predict the pollution level
    prediction = pipeline.predict(X_country)
    
    # Display the prediction
    messagebox.showinfo("Prediction", f"The predicted PM2.5 AQI Value for {country} is: {prediction[0]}")

# Function to plot air pollution by country
def plot_pollution_by_country():
    # Load the dataset
    dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
    cleaned_dataset = clean_dataset(dataset)
    engineered_dataset = feature_engineer(cleaned_dataset)
    
    # Aggregate data by country
    numeric_columns = engineered_dataset.select_dtypes(include=['number']).columns
    country_pollution = engineered_dataset.groupby('Country')[numeric_columns].mean().reset_index()
    
    # Plot the graph
    plt.figure(figsize=(36, 12))  # Increased width to 36 inches
    sns.barplot(x='Country', y='PM2.5 AQI Value', data=country_pollution, palette='viridis')
    plt.xticks(rotation=90, ha='center', fontsize=10)  # Center alignment
    plt.gca().margins(x=0)
    plt.gcf().subplots_adjust(left=0.02, bottom=0.4)  # Reduced left margin
    plt.title('Average PM2.5 AQI Value by Country')
    plt.xlabel('Country')
    plt.ylabel('Average PM2.5 AQI Value')
    plt.tight_layout()  # Tighter layout
    plt.show()

# Function to plot AQI distribution
def plot_aqi_distribution():
    # Load the dataset
    dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
    cleaned_dataset = clean_dataset(dataset)
    
    # Plot AQI distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(cleaned_dataset['PM2.5 AQI Value'], kde=True, bins=30)
    plt.title('PM2.5 AQI Value Distribution')
    plt.xlabel('PM2.5 AQI Value')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot correlation heatmap
def plot_correlation_heatmap():
    # Load the dataset
    dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
    cleaned_dataset = clean_dataset(dataset)
    
    # Select only numeric columns for correlation heatmap
    numeric_data = cleaned_dataset.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    correlation_matrix = numeric_data.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Air Pollution Predictor")

# Country label and entry
ttk.Label(root, text="Country:").grid(row=0, column=0, padx=10, pady=10)
country_entry = ttk.Entry(root)
country_entry.grid(row=0, column=1, padx=10, pady=10)

# Predict button
predict_button = ttk.Button(root, text="Predict", command=lambda: predict_pollution(country_entry.get()))
predict_button.grid(row=1, column=0, columnspan=2, pady=10)

# Plot buttons
plot_button = ttk.Button(root, text="Compare Pollution by Country", command=plot_pollution_by_country)
plot_button.grid(row=2, column=0, columnspan=2, pady=10)

aqi_dist_button = ttk.Button(root, text="Plot AQI Distribution", command=plot_aqi_distribution)
aqi_dist_button.grid(row=3, column=0, columnspan=2, pady=10)

corr_heatmap_button = ttk.Button(root, text="Plot Correlation Heatmap", command=plot_correlation_heatmap)
corr_heatmap_button.grid(row=4, column=0, columnspan=2, pady=10)

# Run the GUI
root.mainloop()
