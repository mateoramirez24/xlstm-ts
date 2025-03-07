# src/ml/models/xlstm_ts/logic.py

import pandas as pd
from ml.models.shared.visualisation import visualise
from ml.models.shared.metrics import calculate_metrics
from ml.models.shared.directional_prediction import evaluate_directional_movement
from ml.models.xlstm_ts.preprocessing import inverse_normalise_data_xlstm
from ml.constants import SEQ_LENGTH_XLSTM
from ml.models.xlstm_ts.xlstm_ts_model import create_xlstm_model
from ml.models.xlstm_ts.training import train_model, evaluate_model

# -------------------------------------------------------------------------------------------
# xLSTM-TS logic
# -------------------------------------------------------------------------------------------

def run_xlstm_ts(train_x, train_y, val_x, val_y, test_x, test_y, scaler, stock, data_type, test_dates, train_y_original=None, val_y_original=None, test_y_original=None):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)

    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)

    # Invert the normalisation for comparison
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)

    # If the original data is provided, use it for the evaluation
    if train_y_original is not None and val_y_original is not None and test_y_original is not None:
        train_y = train_y_original
        val_y = val_y_original
        test_y = test_y_original
    
    test_y = inverse_normalise_data_xlstm(test_y, scaler)

    model_name = 'xLSTM-TS'
    metrics_price = calculate_metrics(test_y, test_predictions, model_name, data_type)

    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=True, dates=test_dates)
    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=False, dates=test_dates)

    train_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, train_x)
    val_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, val_x)

    # Invert the normalisation for comparison
    train_predictions = inverse_normalise_data_xlstm(train_predictions.squeeze(), scaler)
    train_y = inverse_normalise_data_xlstm(train_y, scaler)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    val_predictions = inverse_normalise_data_xlstm(val_predictions.squeeze(), scaler)
    val_y = inverse_normalise_data_xlstm(val_y, scaler)

    true_labels, predicted_labels, metrics_direction = evaluate_directional_movement(train_y, train_predictions, val_y, val_predictions, test_y, test_predictions, model_name, data_type, using_darts=False)
=======
    true_labels, predicted_labels, metrics_direction, predictions_df = evaluate_directional_movement(train_y, train_predictions, val_y, val_predictions, test_y, test_predictions, model_name, data_type, using_darts=False)
>>>>>>> Stashed changes

    metrics_price.update(metrics_direction)

    # Combine data into a DataFrame
    data = {
        'Date': test_dates.tolist()[:-1],
        'Close': [item for sublist in test_y for item in sublist][:-1],
        'Predicted Value': [item for sublist in test_predictions for item in sublist][:-1],
        'True Label': true_labels.tolist(),
        'Predicted Label': predicted_labels.tolist()
    }

<<<<<<< Updated upstream
=======
    data = {
        'Date': test_dates.tolist(),
        'Open': [y[0] for y in test_y],  
        'Close': [y[1] for y in test_y],  
        'Predicted Open': [p[0] for p in test_predictions],  
        'Predicted close': [p[1] for p in test_predictions],
    }
    
>>>>>>> Stashed changes
    results_df = pd.DataFrame(data)

    return results_df, metrics_price, predictions_df


def run_xlstm_ts_1out(train_x, train_y, val_x, val_y, test_x, test_y, scaler, stock, data_type, test_dates, train_y_original=None, val_y_original=None, test_y_original=None):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)
    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)

    # Invert the normalisation for comparison
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)


    # If the original data is provided, use it for the evaluation
    if train_y_original is not None and val_y_original is not None and test_y_original is not None:
        train_y = train_y_original
        val_y = val_y_original
        test_y = test_y_original
    
    test_y = inverse_normalise_data_xlstm(test_y, scaler)

    model_name = 'xLSTM-TS'
    metrics_price = calculate_metrics(test_y, test_predictions, model_name, data_type)

    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=True, dates=test_dates)
    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=False, dates=test_dates)

    train_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, train_x)
    val_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, val_x)

    # Invert the normalisation for comparison
    train_predictions = inverse_normalise_data_xlstm(train_predictions.squeeze(), scaler)

    train_y = inverse_normalise_data_xlstm(train_y, scaler)

    val_predictions = inverse_normalise_data_xlstm(val_predictions.squeeze(), scaler)

    val_y = inverse_normalise_data_xlstm(val_y, scaler)


=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
    true_labels, predicted_labels, metrics_direction, predictions_df = evaluate_directional_movement(train_y, train_predictions, val_y, val_predictions, test_y, test_predictions, model_name, data_type, using_darts=False)

    metrics_price.update(metrics_direction)

    # Combine data into a DataFrame
    data = {
        'Date': test_dates.tolist()[:-1],
        'Close': [item for sublist in test_y for item in sublist][:-1],
        'Predicted Value': [item for sublist in test_predictions for item in sublist][:-1],
        'True Label': true_labels.tolist(),
        'Predicted Label': predicted_labels.tolist()
    }

    data = {
        'Date': test_dates.tolist(),
        'Value': [y for y in test_y],  
        'Predicted Open': [p for p in test_predictions]
    }
    
    results_df = pd.DataFrame(data)

    return results_df, metrics_price, predictions_df
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes


def run_xlstm_ts_1out(train_x, train_y, val_x, val_y, test_x, test_y, scaler, stock, data_type, test_dates, train_y_original=None, val_y_original=None, test_y_original=None):
    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)
    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, train_x, train_y, val_x, val_y)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, test_x)

    # Invert the normalisation for comparison
    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler)


    # If the original data is provided, use it for the evaluation
    if train_y_original is not None and val_y_original is not None and test_y_original is not None:
        train_y = train_y_original
        val_y = val_y_original
        test_y = test_y_original
    
    test_y = inverse_normalise_data_xlstm(test_y, scaler)

    model_name = 'xLSTM-TS'
    metrics_price = calculate_metrics(test_y, test_predictions, model_name, data_type)

    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=True, dates=test_dates)
    visualise(test_y, test_predictions, stock, model_name, data_type, show_complete=False, dates=test_dates)

    train_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, train_x)
    val_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, val_x)

    # Invert the normalisation for comparison
    train_predictions = inverse_normalise_data_xlstm(train_predictions.squeeze(), scaler)

    train_y = inverse_normalise_data_xlstm(train_y, scaler)

    val_predictions = inverse_normalise_data_xlstm(val_predictions.squeeze(), scaler)

    val_y = inverse_normalise_data_xlstm(val_y, scaler)


    true_labels, predicted_labels, metrics_direction, predictions_df = evaluate_directional_movement(train_y, train_predictions, val_y, val_predictions, test_y, test_predictions, model_name, data_type, using_darts=False)

    metrics_price.update(metrics_direction)

    # Combine data into a DataFrame
    data = {
        'Date': test_dates.tolist()[:-1],
        'Close': [item for sublist in test_y for item in sublist][:-1],
        'Predicted Value': [item for sublist in test_predictions for item in sublist][:-1],
        'True Label': true_labels.tolist(),
        'Predicted Label': predicted_labels.tolist()
    }

    data = {
        'Date': test_dates.tolist(),
        'Value': [y for y in test_y],  
        'Predicted Open': [p for p in test_predictions]
    }
    
    results_df = pd.DataFrame(data)

    return results_df, metrics_price, predictions_df
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
