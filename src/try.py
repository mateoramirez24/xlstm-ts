# Ticker (check in Yahoo Finance) and custom stock name
TICKER = '^GSPC' # S&P 500 index
STOCK = 'S&P 500'

# Date range (YYYY-MM-DD) and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d' # daily frequency

FILE_NAME = 'sp500_daily' # custom file name

TRAIN_END_DATE = '2021-01-01'
VAL_END_DATE = '2022-07-01'

from ml.utils.imports import *

from ml.data.download import download_data
from ml.utils.visualisation import plot_data

# Download the data
df = download_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE, freq=FREQ)

# Plot the raw data
plot_data(df, STOCK)

from ml.data.preprocessing import wavelet_denoising, plot_wavelet_denoising

# Apply denoising
df['Close_denoised'] = wavelet_denoising(df['Close'])
df['Noise'] = df['Close'] - df['Close_denoised']

plot_wavelet_denoising(df, STOCK)

from ml.data.preprocessing import process_dates

# Convert the Date column to time zone-naive datetime
df = process_dates(df)

from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm

close_scaled, scaler = normalise_data_xlstm(df['Close'].values)
close_scaled_denoised, scaler_denoised = normalise_data_xlstm(df['Close_denoised'].values)

from ml.models.xlstm_ts.preprocessing import create_sequences

X, y, dates = create_sequences(close_scaled, df.index)

X_denoised, y_denoised, _ = create_sequences(close_scaled_denoised, df.index)

from ml.models.xlstm_ts.preprocessing import split_train_val_test_xlstm

train_X, train_y, train_dates, val_X, val_y, val_dates, test_X, test_y, test_dates = split_train_val_test_xlstm(X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler, STOCK)

train_X_denoised, train_y_denoised, _, val_X_denoised, val_y_denoised, _, test_X_denoised, test_y_denoised, _ = split_train_val_test_xlstm(X_denoised, y_denoised, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_denoised, STOCK)

from ml.models.shared.directional_prediction import *
from ml.models.shared.metrics import *
from ml.models.shared.visualisation import *
from ml.models.darts.darts_models import *
from ml.models.darts.training import *

metrics_accumulator = {}
metrics_accumulator_denoised = {}

from ml.models.xlstm_ts.xlstm_ts_model import *
from ml.models.xlstm_ts.logic import *

plot_architecture_xlstm()

model_name = 'xLSTM-TS'

results_df, metrics = run_xlstm_ts(train_X, train_y, val_X, val_y, test_X, test_y, scaler, STOCK, 'Original', test_dates)

metrics_accumulator[model_name] = metrics

results_denoised_df, metrics_denoised = run_xlstm_ts(train_X_denoised, train_y_denoised, val_X_denoised, val_y_denoised, test_X_denoised, test_y, scaler_denoised, STOCK, 'Denoised', test_dates, train_y, val_y, test_y)

metrics_accumulator_denoised[model_name] = metrics_denoised

from ml.results.results import *

final_results = display_metrics(metrics_accumulator, metrics_accumulator_denoised)

original_data = show_results(final_results, 'Original')
denoised_data = show_results(final_results, 'Denoised')

print(denoised_data)