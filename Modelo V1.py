import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

plt.style.use('fivethirtyeight')

# Ruta del archivo
rutaArchivo = './Datos.xlsx'

# Carga los datos desde el archivo Excel en un DataFrame de pandas
datos = pd.read_excel(rutaArchivo, decimal=',')

# Tratar valores faltantes
datos = datos.dropna()

# Definir las columnas de entrada y salida:
input_columns = ['hPV [cm]', 'hSP [cm]', 'PPV [mbar]',
                 'PSP [mbar]', 'Potencia de la Bomba [hp]']
output_columns = ['% Apertura Presion', '% Apertura Nivel']

# Crear una instancia de MinMaxScaler
scaler = MinMaxScaler()

# Normalizar las columnas de entrada
datos[input_columns] = scaler.fit_transform(datos[input_columns])

# Normalizar las columnas de salida
datos[output_columns] = scaler.fit_transform(datos[output_columns])

# División de conjuntos de datos con TimeSeriesSplit
X_train_list = []
X_val_test_list = []

tscv = TimeSeriesSplit(n_splits=5)  # Número de divisiones
for train_index, test_index in tscv.split(datos):
    X_train_list.append(datos.iloc[train_index])
    X_val_test_list.append(datos.iloc[test_index])

# Combinar los conjuntos de entrenamiento y prueba
X_train = pd.concat(X_train_list)
X_val_test = pd.concat(X_val_test_list)

# Dividir el conjunto de validación y prueba
X_val, X_test = train_test_split(X_val_test, test_size=0.5, shuffle=False)

# Definir la longitud de la secuencia de tiempo
sequence_length = 50  # Ajusta este valor según tus necesidades


def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length][output_columns].values)
    return np.array(X), np.array(y)


# Crear secuencias de entrenamiento
X_train_input, y_train = create_sequences(X_train, sequence_length)

# Crear secuencias de validación
X_val_input, y_val = create_sequences(X_val, sequence_length)

# Crear secuencias de prueba
X_test_input, y_test = create_sequences(X_test, sequence_length)

# Transformar datos en tensores
X_train_tensor = tf.convert_to_tensor(X_train_input, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_val_tensor = tf.convert_to_tensor(X_val_input, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val, dtype=tf.float32)

X_test_tensor = tf.convert_to_tensor(X_test_input, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)


# Crear el modelo
model = Sequential()

# Agregar una capa LSTM
# Disiminuir el número de neuronas para evitar el sobreajuste de 64 a 32.
model.add(LSTM(32, input_shape=(sequence_length, len(input_columns) + 2)))

# Agregar una capa densa para la salida
model.add(Dense(len(output_columns)))

# Definir función de pérdida RMSE


def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# Compilar el modelo
model.compile(loss=rmse_loss, optimizer='adam', metrics=['accuracy'])

# Imprimir un resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train_tensor, y_train_tensor, batch_size=32,
                    epochs=5, validation_data=(X_val_tensor, y_val_tensor))

# Evaluar el rendimiento del modelo en el conjunto de prueba
loss = model.evaluate(X_test_tensor, y_test_tensor)

print("Pérdida en el conjunto de prueba:", loss)

# Graficar los errores en función de las épocas
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Error RMS')
plt.legend()
plt.show()

# Obtener las predicciones del modelo en el conjunto de prueba
y_pred = model.predict(X_test_tensor)

# Desnormalizar las predicciones y los datos de prueba
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test_tensor)

# Crear un DataFrame con los datos predichos y los datos de prueba
df_pred = pd.DataFrame({'Esperado_1': y_test[:, 0], 'Predicho_1': y_pred[:, 0],
                        'Esperado_2': y_test[:, 1], 'Predicho_2': y_pred[:, 1]})

# Guardar los datos en un archivo Excel
df_pred.to_excel('datos_predichos.xlsx', index=False)

# Graficar los resultados de las salidas
# Obtener las salidas predichas
pred_1 = y_pred[:, 0]
pred_2 = y_pred[:, 1]

# Obtener las salidas reales
real_1 = y_test[:, 0]
real_2 = y_test[:, 1]

# Crear una figura con dos subtramas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Graficar la salida 1
ax1.plot(pred_1, label='Predicho', linewidth=2)
ax1.plot(real_1, label='Real', linewidth=2)
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('% Apertura Presión')
ax1.legend()

# Graficar la salida 2
ax2.plot(pred_2, label='Predicho', linewidth=2)
ax2.plot(real_2, label='Real', linewidth=2)
ax2.set_xlabel('Tiempo')
ax2.set_ylabel('% Apertura Nivel')
ax2.legend()

# Ajustar los márgenes y espaciado
fig.tight_layout()

# Mostrar la figura
plt.show()
