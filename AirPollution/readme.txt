# Air Pollution Analysis and Prediction

## Project Overview

The "AirPollution" project aims to predict air pollution levels based on various AQI (Air Quality Index) values using a trained machine learning model. This project includes a graphical user interface (GUI) for easy interaction and visualization of air pollution data.

## Project Structure
AirPollution/
│
├── PROJECT/
│ ├── global air pollution dataset.csv
│ ├── model_pipeline.joblib
│
├── src/
│ ├── init.py
│ ├── data_processing.py
│ ├── model.py
│ └── train_model.py
│
├── main.py
├── requirements.txt
├── README.md


## Installation

1. Clone the repository:
      git clone https://github.com/yourusername/AirPollution.git
      cd AirPollution
   
2. Install the dependencies:
      pip install -r requirements.txt

## Training the Model
3. Ensure the dataset (global air pollution dataset.csv) is placed in the AirPollution/PROJECT directory.
   Run the training script to train the model and save it as model_pipeline.joblib
      python src/train_model.py
## Running the GUI 
4. Run the main script to launch the GUI
      python main.py


