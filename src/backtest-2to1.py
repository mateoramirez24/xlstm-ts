from ml.utils.imports import *
from ml.utils.visualisation import plot_data
from ml.data.preprocessing import wavelet_denoising, plot_wavelet_denoising
from ml.data.preprocessing import process_dates
from ml.models.xlstm_ts.preprocessing import normalise_data_xlstm
from ml.models.xlstm_ts.preprocessing import create_sequences
from ml.models.xlstm_ts.xlstm_ts_model import *
from ml.models.xlstm_ts.logic import *
from ml.results.results import *
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
pd.options.mode.chained_assignment = None

# Ticker (check in Yahoo Finance) and custom stock name
TICKER = 'SPY' #'^GSPC'  S&P 500 index
STOCK = 'S&P 500'

# Date range (YYYY-MM-DD) and frequency
START_DATE = '2000-01-01' #LA ORIGINAL ES CON 2000
END_DATE = '2023-12-31'
FREQ = '1d' # daily frequency

FILE_NAME = 'sp500_daily' # custom file name

TRAIN_END_DATE = '2021-10-01'
VAL_END_DATE = '2023-03-31'
END_EVAL_DATE = datetime(2025, 3, 6)

output_folder = "src/backtesting-2to1"

# Download the data
df = yf.download("SPY", start=START_DATE,end= '2025-03-04', multi_level_index=False, auto_adjust=False)

# Plot the raw data
plot_data(df, STOCK)

##Mensual
"""
# Obtener la fecha actual
hoy = datetime.today()

# Determinar el primer día del mes actual
primer_dia_mes_actual = datetime(hoy.year, hoy.month, 1)

# Iterar sobre los últimos 2 años hasta el primer día del mes actual
val_fechas = []
test_fechas = []
for i in range(24):  # 24 meses (2 años)
    primer_dia_mes = primer_dia_mes_actual - relativedelta(months=i)  # Retroceder mes a mes
    un_anio_atras = primer_dia_mes - relativedelta(years=1)  # Un año atrás
    
    test_fechas.append(primer_dia_mes)
    val_fechas.append(un_anio_atras)

val_fechas.reverse()
test_fechas.reverse()

# Imprimir resultados
for f_actual in test_fechas:
    print(f"Mes: {f_actual.strftime('%Y-%m-%d')}")

for f_pasada in val_fechas:
    print(f"Un año atrás: {f_pasada.strftime('%Y-%m-%d')}")
"""

inputs = ['Open', 'Close']
for input in inputs:
    df[f'{input}_denoised'] = wavelet_denoising(df[f'{input}'])
    df[f'Noise_{input}'] = df[f'{input}'] - df[f'{input}_denoised']

plot_wavelet_denoising(df, STOCK)

# Crear DataFrame con las fechas y predicciones

# Generar el nombre del archivo basado en la fecha de evaluación
nombre_archivo = f"merged/real_data.xlsx"

# Guardar en CSV
ruta_completa = os.path.join(output_folder, nombre_archivo)
df.to_excel(ruta_completa, engine='openpyxl')

print(f"Archivo guardado: {nombre_archivo}")

# Convert the Date column to time zone-naive datetime
df = process_dates(df)

# Punto de inicio: entrenar con datos hasta enero 2023
fecha_inicio_train = datetime(2000, 1, 1)

fecha_inicio_val = datetime(2021, 12, 1)
fecha_fin_train_base = fecha_inicio_val
fecha_fin_val_base = datetime(2022, 12, 1)
fecha_eval_base = datetime(2023, 3, 1)

close_scaled, scaler_close = normalise_data_xlstm(df['Close'].values)
open_scaled, scaler_open = normalise_data_xlstm(df['Open'].values)


# Construir matriz de features combinando Open y Close denoised
features_noised = np.column_stack((open_scaled, close_scaled))

X, y, dates = create_sequences(features_noised, df.index)
dates = pd.to_datetime(dates).squeeze()

while fecha_eval_base < END_EVAL_DATE:


    fecha_fin_train = fecha_fin_train_base - timedelta(days=1)
    fecha_fin_val = fecha_fin_val_base - timedelta(days=1)
    fecha_eval = fecha_eval_base - timedelta(days=1)


    mask_train = (dates <= fecha_fin_train)
    mask_val = (dates > fecha_fin_train) & (dates <= fecha_fin_val)
    mask_test = (dates > fecha_fin_val) & (dates <= fecha_eval)
    mask_traineval = (dates <= fecha_fin_val)


    close_denoised_trainval = wavelet_denoising(df['Close'].iloc[np.where(mask_traineval)].values)
    open_denoised_trainval = wavelet_denoising(df['Open'].iloc[np.where(mask_traineval)].values)

    close_scaled_traineval, scaler_close_traineval = normalise_data_xlstm(
        close_denoised_trainval
    )
    open_scaled_traineval, scaler_open_traineval = normalise_data_xlstm(
        open_denoised_trainval
    )

    df['Close_den'] = df['Close'].copy()
    df['Open_den'] = df['Open'].copy()

    df['Close_den'].iloc[np.where(mask_traineval)] = close_denoised_trainval
    df['Open_den'].iloc[np.where(mask_traineval)] = open_denoised_trainval

    close_scaled_data = scaler_close_traineval.transform(df['Close_den'].values.reshape(-1, 1))
    open_scaled_data = scaler_open_traineval.transform(df['Open_den'].values.reshape(-1, 1))

    features = np.column_stack((open_scaled_data, close_scaled_data))
    X, _, _ = create_sequences(features, df.index)
    _, y, _ = create_sequences(close_scaled_data, df.index)


    X_train = X[mask_train]
    y_train = y[mask_train]
    dates_train = dates[mask_train]

    X_val = X[mask_val]
    y_val = y[mask_val]
    dates_val = dates[mask_val]

    X_test = X[mask_test]
    y_test = y[mask_test]
    dates_test = dates[mask_test.to_numpy()]


    xlstm_stack, input_projection, output_projection = create_xlstm_model(SEQ_LENGTH_XLSTM)
    xlstm_stack, input_projection, output_projection = train_model(xlstm_stack, input_projection, output_projection, X_train, y_train, X_val, y_val)

    test_predictions = evaluate_model(xlstm_stack, input_projection, output_projection, X_test)

    test_predictions = inverse_normalise_data_xlstm(test_predictions.squeeze(), scaler_close_traineval)
    y_test = inverse_normalise_data_xlstm(y_test, scaler_close_traineval)

    # Crear DataFrame con las fechas y predicciones
    df_results = pd.DataFrame({
        'Date': dates_test,
        'Actual_close': y_test.ravel(),
        'Predicted_Close': test_predictions.ravel()
    })

    # Asegurar que 'Date' sea índice
    df_results.set_index('Date', inplace=True)

    # Generar el nombre del archivo basado en la fecha de evaluación
    nombre_archivo = f"predictions/predictions_{fecha_eval.strftime('%m_%Y')}.xlsx"

    # Guardar en CSV
    ruta_completa = os.path.join(output_folder, nombre_archivo)
    df_results.to_excel(ruta_completa, engine='openpyxl')

    print(f"Archivo guardado: {nombre_archivo}")

    fecha_inicio_train += relativedelta(months=3)
    fecha_fin_train_base += relativedelta(months=3)
    fecha_inicio_val += relativedelta(months=3)
    fecha_fin_val_base += relativedelta(months=3)
    fecha_eval_base += relativedelta(months=3)



directorio = "src/backtesting-2to1"
predictions_path = os.path.join(directorio, "predictions")
# Lista para almacenar los DataFrames
dfs = []

# Recorrer todos los archivos en el directorio
for archivo in os.listdir(predictions_path):
    if archivo.endswith(".xlsx"):  # Filtrar solo archivos de Excel
        ruta_archivo = os.path.join(predictions_path, archivo)
        df = pd.read_excel(ruta_archivo)  # Leer archivo Excel
        
        # Convertir la columna de fecha a datetime (ajústala al nombre real de tu columna)
        df["Date"] = pd.to_datetime(df["Date"])  
        
        dfs.append(df)

# Concatenar todos los DataFrames en uno solo
df_final = pd.concat(dfs, ignore_index=True)

# Ordenar por la columna de fecha
df_final = df_final.sort_values(by="Date", ascending=True).reset_index(drop=True)

# Guardar el resultado en un nuevo archivo Excel dentro de src/backtesting
ruta_salida = os.path.join(directorio, "merged/predicciones_completas.xlsx")
df_final.to_excel(ruta_salida, index=False)

print(f"Merge completado. Archivo guardado en: {ruta_salida}")

archivo = ruta_salida  # Reemplaza con la ruta de tu archivo
df = pd.read_excel(archivo)

# Columnas a las que aplicar la operación (ajusta los nombres)
col1, col2 = "Actual_close", "Predicted_Close"

# Calcular signo(diff(columna)) y agregar NaN al final para mantener la longitud
df[f"real_dir"] = np.maximum(0, np.sign(df[col1].diff()))
df[f"pred_dir"] = np.maximum(0, np.sign(df[col2].diff()))
df["Res"] = np.where(df[f"pred_dir"] == df[f"real_dir"], 1, 0)
print(df["Res"].mean())

# Guardar el DataFrame modificado en un nuevo archivo
df.to_excel("src/backtesting-2to1/merged/directions.xlsx", index=False)

print("Archivo guardado con las nuevas columnas.")
