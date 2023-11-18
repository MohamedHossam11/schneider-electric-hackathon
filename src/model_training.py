import pandas as pd
import argparse
import tensorflow as tf

def load_data(file_path):
    # TODO: Load processed data from CSV file
    df = {
    'SP': [9, 100, 250, 150, 900],
    'UK': [12, 39, 1982, 11, 22],
    'DE': [1, 100, 250, 150, 9000],
    'DK': [9000, 100, 250, 150, 900],
    'HU': [9, 100000, 250, 150, 900],
    'SE': [9, 100, 250, 150, 900],
    'IT': [9, 100, 250, 150000, 900],
    'PO': [9, 100, 250, 150, 900],
    'NL': [91, 100, 25000, 150, 900]
}
    # df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = add_labels(df)
    print(df)
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
        max_column_name = row.index[1:][row.values[1:].argmax()]
        labels.append(country_map[max_column_name])
    
    data_frame['label'] = labels
    
    return data_frame

def split_data(df):
    # TODO: Split data into training and validation sets (the test set is already provided in data/test_data.csv)
    return X_train, X_val, y_train, y_val

def train_model(X_train, y_train):
    # TODO: Initialize your model and train it

    return model

def save_model(model, model_path):
    # TODO: Save your trained model
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

def main(input_file, model_file):
    df = load_data(input_file)
    X_train, X_val, y_train, y_val = split_data(df)
    model = train_model(X_train, y_train)
    save_model(model, model_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file)