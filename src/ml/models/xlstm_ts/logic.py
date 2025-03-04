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
    binary_test_predictions = (test_predictions >= 0.5).float()

    # If the original data is provided, use it for the evaluation
    if train_y_original is not None and val_y_original is not None and test_y_original is not None:
        train_y = train_y_original
        val_y = val_y_original
        test_y = test_y_original


    model_name = 'xLSTM-TS'

    train_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, train_x)
    val_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, val_x)

    binary_train_predictions = (train_predictions >= 0.5).float()
    binary_val_predictions = (val_predictions >= 0.5).float()

    accur_train = (binary_train_predictions == train_y).float().mean()
    accur_val = (binary_val_predictions == val_y).float().mean()
    accur_test = (binary_test_predictions == test_y).float().mean()

    return accur_train, accur_val, accur_test
