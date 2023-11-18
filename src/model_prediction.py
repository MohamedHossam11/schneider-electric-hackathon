import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.metrics import mean_absolute_error



def load_data(file_path):
    # TODO: Load test data from CSV file
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    return df

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(df, model):
    y_pred = model.predict(df)
    y_pred =y_pred.flatten()
    y_val = pd.read_csv('data/testing_labels.csv')
    print(y_val)
    mae = mean_absolute_error(y_val, y_pred)
    accuracy_percentage = (1 - (mae / np.mean(y_val))) * 100
    print(accuracy_percentage)
    return y_pred

def save_predictions(predictions, predictions_file):
    # TODO: Save predictions to a JSON file
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prediction script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='data/test_data.csv', 
        help='Path to the test data file to make predictions'
    )
    parser.add_argument(
        '--model_file', 
        type=str, 
        default='models/model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='predictions/predictions.json', 
        help='Path to save the predictions'
    )
    return parser.parse_args()

def main(input_file, model_file, output_file):
    df = load_data(input_file)
    model = load_model(model_file)
    predictions = make_predictions(df, model)
    save_predictions(predictions, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.model_file, args.output_file)