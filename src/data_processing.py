import argparse
import pandas as pd
import os

countries = ['DE', 'DK', 'HU', 'IT', 'NE', 'PO', 'SE', 'SP', 'UK']

def load_data(file_path):
    # TODO: Load data from CSV file
    csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
    file_path_temp = file_path
    dataframe_list_gen = [[] for _ in range(len(countries))]
    dataframe_list_load = [[] for _ in range(len(countries))]

    for file in csv_files:
        if file == 'test.csv': continue
        name_split = file.split('_')
        if(name_split[0] == 'gen'):
            file_path = os.path.join(file_path, file)
            df = pd.read_csv(file_path)
            index_country = countries.index(name_split[1])
            dataframe_list_gen[index_country].append(df)
            file_path = file_path_temp
        else:
            file_path = os.path.join(file_path, file)
            df = pd.read_csv(file_path)
            index_country = countries.index(name_split[1].split('.')[0])
            dataframe_list_load[index_country].append(df)
            file_path = file_path_temp

    index = 0
    for gen_country in dataframe_list_gen:
        df = pd.concat(gen_country, ignore_index=True)
        df['StartTime'] = df['StartTime'].str.rstrip('Z')
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df.set_index('StartTime', inplace=True)
        resampled_generation = df['quantity'].resample('H').sum()
        resampled_generation_df = resampled_generation.to_frame()
        resampled_generation_df['country'] = countries[index]
        resampled_generation_df.to_csv(countries[index]  + 'gen.csv')
        index += 1

    index = 0
    merged_dfs = []
    for load_country in dataframe_list_load:
        df = pd.concat(load_country, ignore_index=True)
        df['StartTime'] = df['StartTime'].str.rstrip('Z')
        df['StartTime'] = pd.to_datetime(df['StartTime'])
        df.set_index('StartTime', inplace=True)
        resampled_consumption = df['Load'].resample('H').sum()
        resampled_consumption_df = resampled_consumption.to_frame()
        resampled_consumption_df['country'] = countries[index]

        
        resampled_generation_df = pd.read_csv(countries[index] + 'gen.csv')
        resampled_generation_df['StartTime'] = pd.to_datetime(resampled_generation_df['StartTime'])
        merged_df = pd.merge(resampled_generation_df, resampled_consumption_df, on='StartTime')
        merged_dfs.append(merged_df)
        print(merged_df)
        index += 1
    df = pd.concat(merged_dfs)

    return df

def clean_data(df):

    df_clean = df
    return df_clean

def preprocess_data(df):
    df['surplus'] = df['quantity'] - df['Load']
    df = df.drop(columns=['quantity', 'country_x', 'Load'], axis=1)
    df = df.rename(columns= {"country_y": "country"})
    df = df.pivot_table(index='StartTime', columns='country', values='surplus', aggfunc='sum')

    df.reset_index(inplace=True)
    return df

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
    df.to_csv(output_file)
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)