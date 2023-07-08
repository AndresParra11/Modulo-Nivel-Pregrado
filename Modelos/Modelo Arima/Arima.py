import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Configurar el estilo de las gráficas
plt.style.use('fivethirtyeight')

# Ruta del archivo
rutaArchivo = './Datos.xlsx'

# Carga los datos desde el archivo Excel en un DataFrame de pandas
datos = pd.read_excel(rutaArchivo, decimal=',')

# Tratar valores faltantes
datos = datos.dropna()

# Definir las columnas de entrada y salida:
input_columns = ['hPV [cm]', 'hSP [cm]', 'PPV [mbar]', 'PSP [mbar]', 'Frecuencia de la Bomba [Hz]']
output_columns = ['% Apertura Presion', '% Apertura Nivel']

def Escalizado(datos, columna, rango_original, rango_deseado):
    datos[columna] = (datos[columna] - rango_original[0]) / (rango_original[1] - rango_original[0]) * (
            rango_deseado[1] - rango_deseado[0]) + rango_deseado[0]
    return datos[columna]

# Escalizar los datos de las variables de entrada
datos['hPV [cm]'] = Escalizado(datos, 'hPV [cm]', [0, 60], [0, 1])
datos['hSP [cm]'] = Escalizado(datos, 'hSP [cm]', [0, 60], [0, 1])
datos['PPV [mbar]'] = Escalizado(datos, 'PPV [mbar]', [0, 600], [0, 1])
datos['PSP [mbar]'] = Escalizado(datos, 'PSP [mbar]', [0, 600], [0, 1])
datos['Frecuencia de la Bomba [Hz]'] = Escalizado(datos, 'Frecuencia de la Bomba [Hz]', [0, 60], [0, 1])

# Escalizar los datos de las variables de salida
datos['% Apertura Presion'] = Escalizado(datos, '% Apertura Presion', [0, 100], [0, 1])
datos['% Apertura Nivel'] = Escalizado(datos, '% Apertura Nivel', [0, 100], [0, 1])

# División de conjuntos de datos con TimeSeriesSplit
X_train_list = []
X_val_test_list = []

# Dividir el conjunto de datos en 5 partes
tscv = TimeSeriesSplit(n_splits=5)  # Número de divisiones
for train_index, test_index in tscv.split(datos):
    X_train_list.append(datos.iloc[train_index])
    X_val_test_list.append(datos.iloc[test_index])

# Unir el conjunto de entrenamiento.
X_train = pd.concat(X_train_list)
X_val_test = pd.concat(X_val_test_list)

# Dividir el conjunto de validación y prueba de forma equitativa y sin barajear.
X_val, X_test = train_test_split(X_val_test, test_size=0.5, shuffle=False)

# Definir la longitud de la secuencia de tiempo
sequence_length = 10  # Ajusta este valor según tus necesidades

# Definir una función para crear secuencias de longitud sequence_length
def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i + sequence_length][input_columns].values)
        y.append(data.iloc[i + sequence_length][output_columns].values)
    return np.array(X), np.array(y)

# Crear secuencias de entrenamiento
X_train_input, y_train = create_sequences(X_train, sequence_length)

# Crear secuencias de validación
X_val_input, y_val = create_sequences(X_val, sequence_length)

# Crear secuencias de prueba
X_test_input, y_test = create_sequences(X_test, sequence_length)

# Crear un DataFrame con los datos de entrenamiento
df_train = pd.DataFrame(X_train_input[:, -1, :], columns=input_columns)

# Crear un DataFrame con los datos de validación
df_val = pd.DataFrame(X_val_input[:, -1, :], columns=input_columns)

# Crear un DataFrame con los datos de prueba
df_test = pd.DataFrame(X_test_input[:, -1, :], columns=input_columns)

# Entrenar y evaluar el modelo ARIMA para cada columna de salida
for column in output_columns:
    print(f"Entrenando y evaluando el modelo ARIMA para '{column}'...")

    # Obtener los datos de entrenamiento y validación para la columna actual
    y_train_column = y_train[:, output_columns.index(column)]
    y_val_column = y_val[:, output_columns.index(column)]

    # Crear el modelo ARIMA
    model = ARIMA(y_train_column, order=(1, 1, 1))  # Ajusta los parámetros del modelo ARIMA según tus necesidades

    # Ajustar el modelo ARIMA
    model_fit = model.fit()

    # Realizar predicciones en los datos de validación
    predictions_val = model_fit.predict(start=len(y_train_column), end=len(y_train_column) + len(y_val_column) - 1)

    # Realizar predicciones en los datos de prueba
    predictions_test = model_fit.predict(start=len(y_train_column) + len(y_val_column),
                                         end=len(y_train_column) + len(y_val_column) + len(y_test[:, 0]) - 1)*100
    

    # Calcular el error RMSE en los datos de prueba
    rmse_test = np.sqrt(np.mean((predictions_test - 100*y_test[:, output_columns.index(column)]) ** 2))
    print(f"Error RMSE en los datos de prueba para '{column}': {rmse_test}")

    # Graficar las predicciones y los valores reales para los datos de prueba.
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(y_test[:, output_columns.index(column)])), 100*y_test[:, output_columns.index(column)], label='Real')
    plt.plot(range(len(y_test[:, output_columns.index(column)])), predictions_test, label='Predicción')
    plt.title(f"Pred. ARIMA para '{column}' - Prueb.")
    plt.xlabel('Tiempo')
    plt.ylabel(column)
    plt.legend()
    plt.savefig(f'grafica_predicciones_{column}_test.png')
    plt.show()
