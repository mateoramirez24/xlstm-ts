# src/ml/results/results.py

import pandas as pd
import matplotlib.pyplot as plt

DATA_TYPE_COLNAME = 'data_type'

def display_metrics(metrics_accumulator, metrics_accumulator_denoised):
    # Convert metrics dictionaries to DataFrames
    results_df = pd.DataFrame.from_dict(metrics_accumulator, orient="index")
    results_df_denoised = pd.DataFrame.from_dict(metrics_accumulator_denoised, orient="index")

    # Add a column to indicate the data type
    results_df[DATA_TYPE_COLNAME] = 0 # Original Data
    results_df_denoised[DATA_TYPE_COLNAME] = 1 # Denoised Data

    # Combine the DataFrames
    combined_df = pd.concat([results_df, results_df_denoised])

    # Move DATA_TYPE_COLNAME to the first column
    cols = combined_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index(DATA_TYPE_COLNAME)))
    combined_df = combined_df[cols]

    return combined_df

# Function to add percentage signs to appropriate columns
def _format_percentage(df):
    # Columns that should be formatted as percentages
    percentage_columns = ['Test Accuracy', 'Train Accuracy', 'Validation Accuracy', 'MAPE', 'Recall', 'Precision (Rise)', 'Precision (Fall)', 'F1 Score']
    for column in percentage_columns:
        df[column] = df[column].apply(lambda x: f"{x:.2f}%")
    return df

def show_results(results, data_type):
    if data_type == 'Original':
        data_type_code = 0
    elif data_type == 'Denoised':
        data_type_code = 1

    final_results_sorted = results.sort_values(by=DATA_TYPE_COLNAME, ascending=False)
    
    data_results = final_results_sorted[final_results_sorted[DATA_TYPE_COLNAME] == data_type_code].drop(columns=[DATA_TYPE_COLNAME]).round(2)

    # Format the original and denoised data
    data = _format_percentage(data_results)

    return data

def save_results(results_original, results_denoised, dataset_name):
    # Save DataFrame as CSV
    csv_filename = f'xlstm_predictions_{dataset_name}.csv'
    results_original.to_csv(csv_filename, index=False)

    # Save DataFrame as CSV
    csv_filename = f'xlstm_predictions_{dataset_name}_denoised.csv'
    results_denoised.to_csv(csv_filename, index=False)
