import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.impute import SimpleImputer
import joblib

# Function to load the data
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to clean the data
def clean_dataset(data):
    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = data.select_dtypes(exclude=['float64', 'int64']).columns
    
    # Impute numeric columns
    imputer = SimpleImputer(strategy='mean')
    data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)
    
    # Handle non-numeric columns (e.g., fill missing values with the most frequent value)
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')
    data_non_numeric_imputed = pd.DataFrame(imputer_non_numeric.fit_transform(data[non_numeric_cols]), columns=non_numeric_cols)
    
    # Combine the imputed numeric and non-numeric data
    data_imputed = pd.concat([data_numeric_imputed, data_non_numeric_imputed], axis=1)
    
    # Ensure the column order is the same as the original data
    data_imputed = data_imputed[data.columns]
    
    return data_imputed

# Function to perform feature engineering
def feature_engineer(data):
    # Create a feature for geographic regions based on the 'Country' column
    region_map = {
        'North America': ['United States', 'Canada', 'Mexico', 'United States of America', 'Haiti', 'El Salvador', 'Guatemala', 'Cuba', 'Panama', 'Dominican Republic', 'Honduras', 'Nicaragua', 'Costa Rica', 'Jamaica', 'Barbados', 'Saint Lucia', 'Trinidad and Tobago', 'Saint Kitts and Nevis'],
        'Europe': ['United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Poland', 'Russian Federation', 'Greece', 'Portugal', 'Sweden', 'Belgium', 'Netherlands', 'Republic of North Macedonia', 'Romania', 'Finland', 'United Kingdom of Great Britain and Northern Ireland', 'Latvia', 'Bulgaria', 'Ireland', 'Turkey', 'Switzerland', 'Denmark', 'Hungary', 'Kazakhstan', 'Lithuania', 'Azerbaijan', 'Armenia', 'Ukraine', 'Serbia', 'Slovakia', 'Bosnia and Herzegovina', 'Czechia', 'Austria', 'Croatia', 'Republic of Moldova', 'Belarus', 'Norway', 'Malta', 'Slovenia', 'Cyprus', 'Montenegro', 'Iceland', 'Andorra', 'Luxembourg', 'Aruba', 'Monaco'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Indonesia', 'Thailand', 'Malaysia', 'Philippines', 'Vietnam', 'Pakistan', 'Viet Nam', 'Iran (Islamic Republic of)', 'Myanmar', 'Israel', 'Bangladesh', 'Kyrgyzstan', 'Saudi Arabia', 'Lebanon', 'Nepal', 'Sri Lanka', 'United Arab Emirates', 'Tajikistan', 'Iraq', 'Syrian Arab Republic', 'Lao People\'s Democratic Republic', 'Bhutan', 'Yemen', 'Turkmenistan', 'Kuwait', 'Georgia', 'Jordan', 'Singapore', 'Maldives', 'Bahrain', 'State of Palestine'],
        'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru', 'Venezuela', 'Uruguay', 'Paraguay', 'Bolivia', 'Ecuador', 'Guyana', 'Suriname'],
        'Africa': ['Nigeria', 'South Africa', 'Egypt', 'Kenya', 'Ethiopia', 'Morocco', 'Algeria', 'Ghana', 'Uganda', 'Tanzania', 'Republic of North Macedonia', 'South Sudan', 'Democratic Republic of the Congo', 'Cameroon', 'Côte d\'Ivoire', 'Madagascar', 'Sudan', 'Congo', 'Mauritius', 'Malawi', 'Benin', 'Sierra Leone', 'Namibia', 'Senegal', 'Lesotho', 'Zimbabwe', 'Angola', 'Kingdom of Eswatini', 'Afghanistan', 'Uzbekistan', 'Zambia', 'Rwanda', 'Botswana', 'Burundi', 'Central African Republic', 'Niger', 'Mali', 'Burkina Faso', 'Cabo Verde', 'Mozambique', 'Mauritania', 'Guinea-Bissau', 'Eritrea', 'Gabon', 'Liberia', 'Togo', 'Comoros', 'Equatorial Guinea', 'Palau', 'Seychelles'],
        'Oceania': ['Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu']
    }   
    
    def map_region(country):
        for region, countries in region_map.items():
            if country in countries:
                return region
        return 'Other'
    
    if 'Country' in data.columns:
        data['region'] = data['Country'].apply(map_region)
    else:
        raise KeyError("No 'Country' column found in the dataset.")
    
    return data

# Function to predict air pollution
def predict_pollution(country, city):
    # Load the dataset and the model
    dataset = load_dataset('C:\\Users\\antos\\OneDrive\\Desktop\\CODING\\PROJECT\\global air pollution dataset.csv')
    cleaned_dataset = clean_dataset(dataset)
    engineered_dataset = feature_engineer(cleaned_dataset)
    
    # Select the relevant row for prediction
    input_data = engineered_dataset[(engineered_dataset['Country'] == country) & (engineered_dataset['City'] == city)]
    
    if input_data.empty:
        messagebox.showerror("Error", "No data available for the selected country and city.")
        return
    
    # Load the trained model
    model = joblib.load('model.joblib')
    
    # Predict the pollution level
    prediction = model.predict(input_data.drop(columns=['Country', 'City', 'region']))
    
    # Display the prediction
    messagebox.showinfo("Prediction", f"The predicted PM2.5 AQI Value for {city}, {country} is: {prediction[0]}")

# GUI setup
root = tk.Tk()
root.title("Air Pollution Predictor")

# Country label and entry
ttk.Label(root, text="Country:").grid(row=0, column=0, padx=10, pady=10)
country_entry = ttk.Entry(root)
country_entry.grid(row=0, column=1, padx=10, pady=10)

# City label and entry
ttk.Label(root, text="City:").grid(row=1, column=0, padx=10, pady=10)
city_entry = ttk.Entry(root)
city_entry.grid(row=1, column=1, padx=10, pady=10)

# Predict button
predict_button = ttk.Button(root, text="Predict", command=lambda: predict_pollution(country_entry.get(), city_entry.get()))
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

# Run the GUI
root.mainloop()
