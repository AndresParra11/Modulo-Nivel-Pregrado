# Importar librerías
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configurar el estilo de las gráficas
plt.style.use('fivethirtyeight')

# Ruta del archivo
rutaArchivo = './Datos.xlsx'

# Carga los datos desde el archivo Excel en un DataFrame de pandas
datos = pd.read_excel(rutaArchivo, decimal=',')

# Tratar valores faltantes
datos = datos.dropna()

# Definir las columnas de entrada y salida:
input_columns = ['hPV [cm]', 'hSP [cm]', 'PPV [mbar]',
                 'PSP [mbar]', 'Frecuencia de la Bomba [Hz]']
output_columns = ['% Apertura Presion', '% Apertura Nivel']


def Escalizado(datos, columna, rango_original, rango_deseado):
    datos[columna] = (datos[columna] - rango_original[0]) / (rango_original[1] -
                                                             rango_original[0]) * (rango_deseado[1] - rango_deseado[0]) + rango_deseado[0]

    return datos[columna]


# Escalizar los datos de las variables de entrada
datos['hPV [cm]'] = Escalizado(datos, 'hPV [cm]', [0, 60], [0, 1])
datos['hSP [cm]'] = Escalizado(datos, 'hSP [cm]', [0, 60], [0, 1])
datos['PPV [mbar]'] = Escalizado(datos, 'PPV [mbar]', [0, 600], [0, 1])
datos['PSP [mbar]'] = Escalizado(datos, 'PSP [mbar]', [0, 600], [0, 1])
datos['Frecuencia de la Bomba [Hz]'] = Escalizado(
    datos, 'Frecuencia de la Bomba [Hz]', [0, 60], [0, 1])

# Escalizar los datos de las variables de salida
datos['% Apertura Presion'] = Escalizado(
    datos, '% Apertura Presion', [0, 100], [0, 1])
datos['% Apertura Nivel'] = Escalizado(
    datos, '% Apertura Nivel', [0, 100], [0, 1])

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos[input_columns], datos[output_columns], test_size=0.2, random_state=200)

# Transformar datos en tensores
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Crear el modelo de Bosque Aleatorio
model = RandomForestRegressor(n_estimators=100, random_state=200)

# Entrenar el modelo
model.fit(X_train_tensor, y_train_tensor)

# Obtener las predicciones del modelo en el conjunto de prueba
y_pred = model.predict(X_test_tensor)

# Evaluar el modelo en el conjunto de prueba
score = model.score(X_test_tensor, y_pred)
print("Coeficiente de determinación R^2:", score)

# Calcular el RMSE en el conjunto de prueba
rmse = np.sqrt(mean_squared_error(y_test_tensor, y_pred))
print("RMSE en el conjunto de prueba:", rmse*100, "%.")

# Calcular el RMSE para cada salida
rmse_1 = np.sqrt(mean_squared_error(y_test_tensor[:, 0], y_pred[:, 0]))
rmse_2 = np.sqrt(mean_squared_error(y_test_tensor[:, 1], y_pred[:, 1]))

# Imprimir los resultados
print("RMSE para Apertura VC de Presión:", rmse_1*100, "%.")
print("RMSE para Apertura VC de Nivel:", rmse_2*100, "%.")

# Desnormalizar las predicciones y los datos de prueba. Limitar a valores entre 0 y 100
y_pred = y_pred * 100
y_test = y_test_tensor * 100

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Graficar la salida 1
ax1.plot(pred_1, label='Predicho', linewidth=1)
ax1.plot(real_1, label='Real', linewidth=1)
ax1.set_xlabel('Tiempo [s]')
ax1.set_ylabel('Apertura VC de Presión [%]')
ax1.legend()

# Graficar la salida 2
ax2.plot(pred_2, label='Predicho', linewidth=1)
ax2.plot(real_2, label='Real', linewidth=1)
ax2.set_xlabel('Tiempo [s]')
ax2.set_ylabel('Apertura VC de Nivel [%]')
ax2.legend()

# Ajustar los márgenes y espaciado
fig.tight_layout()

plt.savefig('grafica_predicciones.png')
# Mostrar la figura
plt.show()
