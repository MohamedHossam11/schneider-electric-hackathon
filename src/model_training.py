import pandas as pd
import argparse
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
import joblib


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = add_labels(df)
    return df

def add_labels(data_frame):
    country_map = {
        'SP': 0, # Spain
        'UK': 1, # United Kingdom
        'DE': 2, # Germany
        'DK': 3, # Denmark
        'HU': 5, # Hungary
        'SE': 4, # Sweden
        'IT': 6, # Italy
        'PO': 7, # Poland
        'NL': 8 # Netherlands
    }

    labels = []
    
    for index, row in data_frame.iterrows():
        non_empty_values = np.array([value for value in row.values[1:] if pd.notna(value) and value != ''])
        max_column_name = row.index[1:][np.argmax(non_empty_values)]
        labels.append(country_map[max_column_name])
    
    data_frame['label'] = labels
    
    return data_frame

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    df = df.fillna(0.0)
    df['StartTime'] = pd.to_datetime(df['StartTime']).astype(int) // 10**9
    X = df.iloc[:, :-1]
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train, X_val, y_val):
    # TODO: Initialize your model and train it
    model = Sequential()
    model.add(Conv1D(filters=50, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu', padding='same'))
    model.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    return model

def save_model(model, model_path):

    # TODO: Save your trained model
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Model training script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to the processed data file to train the model'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl', 
        help='Path to save the trained model'
    )
    return parser.parse_args()

def main(input_file, model_file = './models'):
    df = load_data('./data/processed_data.csv')
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train, X_val, y_val)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)