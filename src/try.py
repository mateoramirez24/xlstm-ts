# Ticker (check in Yahoo Finance) and custom stock name
TICKER = '^GSPC' # S&P 500 index
STOCK = 'S&P 500'

# Date range (YYYY-MM-DD) and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d' # daily frequency

FILE_NAME = 'sp500_daily' # custom file name

TRAIN_END_DATE = '2024-09-30'
VAL_END_DATE = '2024-12-31'

import matplotlib.pyplot as plt
from ml.utils.imports import *

from ml.data.download import download_data
from ml.utils.visualisation import plot_data

df = yf.download("SPY", start=START_DATE,end= None, multi_level_index=False, auto_adjust=False)

# Plot the raw data
plot_data(df, STOCK)

from ml.data.preprocessing import wavelet_denoising

df['Close_denoised'] = wavelet_denoising(df['Close'])
df["Open_denoised"] = wavelet_denoising(df['Open'])
df["diff_day"] = df['Close_denoised'] - df['Open_denoised']
df["diff_day_noised"] = df['Close'] - df['Open']

from ml.data.preprocessing import process_dates

df = process_dates(df)

from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm

diff_day_scaled, scaler_diff_day = normalise_data_xlstm(df['diff_day'].values) #este normalizer me tira todo pa positivo.

from ml.models.xlstm_ts.preprocessing import create_sequences
"""
X, _, dates = create_sequences(diff_day_scaled, df.index)
_, y, dates = create_sequences(df['diff_day'].values, df.index)
print(y)
_, y_real, _ = create_sequences(df['diff_day_noised'].values, df.index)

"""

from ml.models.xlstm_ts.preprocessing import split_train_val_test_xlstm

train_X, train_y, train_dates, val_X, val_y, val_dates, test_X, test_y, test_dates = split_train_val_test_xlstm(X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_diff_day, STOCK)

_, train_y_real, _, _, val_y_real, _, _, test_y_real, _ = split_train_val_test_xlstm(X, y_real, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_diff_day, STOCK)


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

accur_train, accur_val, accur_test = run_xlstm_ts(train_X, train_y, val_X, val_y, test_X, test_y, scaler_diff_day, STOCK, 'Original', test_dates, train_y_real, val_y_real, test_y_real)

print(f"Train Accuracy = {accur_train}")
print(f"Val Accuracy = {accur_val}")
print(f"Test Accuracy = {accur_test}")