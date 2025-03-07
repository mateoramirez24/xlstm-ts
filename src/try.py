<<<<<<< Updated upstream
=======
from ml.utils.imports import *
from ml.utils.visualisation import plot_data
from ml.data.preprocessing import wavelet_denoising, plot_wavelet_denoising
from ml.data.preprocessing import process_dates
from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm
from ml.models.xlstm_ts.preprocessing import create_sequences
from ml.models.xlstm_ts.preprocessing import split_train_val_test_xlstm
from ml.models.xlstm_ts.xlstm_ts_model import *
from ml.models.xlstm_ts.logic import *
from ml.results.results import *

# Ticker (check in Yahoo Finance) and custom stock name
TICKER = 'SPY' #'^GSPC'  S&P 500 index
STOCK = 'S&P 500'

# Date range (YYYY-MM-DD) and frequency
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
FREQ = '1d' # daily frequency

FILE_NAME = 'sp500_daily' # custom file name

TRAIN_END_DATE = '2021-10-01'
VAL_END_DATE = '2023-03-31'

# Download the data
df = yf.download("SPY", start=START_DATE,end= '2025-03-04', multi_level_index=False, auto_adjust=False)

# Plot the raw data
plot_data(df, STOCK)

# Apply denoising
df['Close_denoised'] = wavelet_denoising(df['Close'])
df['Noise_Close'] = df['Close'] - df['Close_denoised']

df['Open_denoised'] = wavelet_denoising(df['Open'])
df['Noise_Open'] = df['Open'] - df['Open_denoised']

plot_wavelet_denoising(df, STOCK)

# Convert the Date column to time zone-naive datetime
df = process_dates(df)

close_scaled, scaler_close = normalise_data_xlstm(df['Close'].values)
open_scaled, scaler_open = normalise_data_xlstm(df['Open'].values)


close_scaled_denoised, scaler_denoised_close = normalise_data_xlstm(df['Close_denoised'].values)
open_scaled_denoised, scaler_denoised_open = normalise_data_xlstm(df['Open_denoised'].values)

# Construir matriz de features combinando Open y Close denoised
features_noised = np.column_stack((open_scaled, close_scaled))
features_denoised = np.column_stack((open_scaled_denoised, close_scaled_denoised))

X, _, dates = create_sequences(features_noised, df.index)
_, y, _ = create_sequences(close_scaled, df.index)

X_denoised, _, _ = create_sequences(features_denoised, df.index)
_, y_denoised, _ = create_sequences(close_scaled_denoised, df.index)

train_X, train_y, train_dates, val_X, val_y, val_dates, test_X, test_y, test_dates = split_train_val_test_xlstm(X, y, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_open, STOCK)

train_X_denoised, train_y_denoised, _, val_X_denoised, val_y_denoised, _, test_X_denoised, test_y_denoised, _ = split_train_val_test_xlstm(X_denoised, y_denoised, dates, TRAIN_END_DATE, VAL_END_DATE, scaler_denoised_open, STOCK)



metrics_accumulator = {}
metrics_accumulator_denoised = {}

plot_architecture_xlstm()

model_name = 'xLSTM-TS'

results_df, metrics, predictions_df = run_xlstm_ts_1out(train_X, train_y, val_X, val_y, test_X, test_y, scaler_close, STOCK, 'Original', test_dates)

metrics_accumulator[model_name] = metrics

results_denoised_df, metrics_denoised, predictions_denoised_df = run_xlstm_ts_1out(train_X_denoised, train_y_denoised, val_X_denoised, val_y_denoised, test_X_denoised, test_y_denoised, scaler_denoised_close, STOCK, 'Denoised', test_dates, train_y, val_y, test_y)

metrics_accumulator_denoised[model_name] = metrics_denoised

final_results = display_metrics(metrics_accumulator, metrics_accumulator_denoised)

dates_index = list(train_dates[:-1]) + list(val_dates[:-1]) + list(test_dates[:-1]) #Acá la fecha es el día base. Es decir, en cada fecha dice si el open va a subir o a bajar al día siguiente
                                                                                    #Es como si lo fuera a correr después de las 4pm del día de la fecha, pronostica, y pilas que pone la dirección pero es respecto a sus predicciones. El modelo no compara si subió o bajo respecto al open/close real, sino que compara rspecto al que él mismo pronosticó (esto hay que revisarlo).


#pareciera que entre más lenght le ponga, más acercados están los datos de los reales, pero como realmente el modelo construye su propia secuencia de datos, el up y down los sigue marcando muy bien.
# Asignar el índice antes de guardar
predictions_df.index = dates_index
predictions_denoised_df.index = dates_index

with pd.ExcelWriter("predicciones.xlsx") as writer:
    predictions_df.to_excel(writer, sheet_name='Original')
    predictions_denoised_df.to_excel(writer, sheet_name="Denoised")

print("Predicciones guardadas en 'predicciones.xlsx'")


#original_data = show_results(final_results, 'Original')
#denoised_data = show_results(final_results, 'Denoised')

#print(denoised_data)
>>>>>>> Stashed changes
