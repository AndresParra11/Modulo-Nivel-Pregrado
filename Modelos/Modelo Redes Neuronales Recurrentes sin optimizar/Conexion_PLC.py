# -*- coding: utf-8 -*-

# Importación de librerías
import pandas as pd
import numpy as np
from asyncua import Client, ua
import h5py
import logging
import asyncio
import tensorflow as tf


# Definir la función de pérdida personalizada
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Registrar la función de pérdida personalizada
tf.keras.utils.get_custom_objects().update({'rmse_loss': rmse_loss})

# Configuración básica del registro de eventos
logging.basicConfig(level=logging.WARNING)

# Obtener el objeto logger para el módulo 'asyncua'
_logger = logging.getLogger('asyncua')

# URL para la conexión OPC UA
url = "opc.tcp://127.0.0.1:49320"

# Definir la ruta del modelo LSTM entrenado
ruta_modelo = 'lstm_model_2.h5'

# Cargar el modelo entrenado
with h5py.File(ruta_modelo, 'r') as file:
    model_cargado = tf.keras.models.load_model(file)

# Crea una función para procesar las variables de entrada y obtener las variables de salida predichas
def process_variables(variables_input):

    # Transformar los datos en tensores
    variables_input_tensor = tf.convert_to_tensor(
        variables_input, dtype=tf.float32)

    # Obtener las variables de salida predichas
    variables_output = model_cargado.predict(variables_input_tensor)

    return variables_output
  
def Escalizado (datos,columna, rango_original, rango_deseado): 
  datos[columna] = (datos[columna] - rango_original[0]) / (rango_original[1] - rango_original[0]) * (rango_deseado[1] - rango_deseado[0]) + rango_deseado[0]
  
  return datos[columna]

# Definición de la función principal
async def main():
    # Crear una instancia del cliente OPC UA
    client = Client(url=url)

    try:
        # Intentar establecer la conexión al servidor OPC UA
        await client.connect()
    except Exception as e:
        # En caso de error al conectar, imprimir el mensaje de excepción y salir de la función
        print(e)
        return

# ------------------------------ Cliente -------------------------------------#
    # Buffer FIFO Sliding Window para almacenar las variables de entrada
    sequence_length = 10
    buffer = []

    while True:
        # Variables de entrada entregadas por el PLC
        nivel_sp = await client.get_node("ns=2;s=Nivel.PLC Nivel.SP Nivel").get_value()
        nivel_pv = await client.get_node("ns=2;s=Nivel.PLC Nivel.Nivel").get_value()
        presion_sp = await client.get_node("ns=2;s=Nivel.PLC Nivel.SP Presión").get_value()
        presion_pv = await client.get_node("ns=2;s=Nivel.PLC Nivel.Presión").get_value()
        frec_bomba = await client.get_node("ns=2;s=Nivel.PLC Nivel.Frec Bomba").get_value()

        # Variables de entrada
        variables_input = [nivel_pv, nivel_sp,
                           presion_pv, presion_sp, frec_bomba]

        # Agregar variables de entrada al buffer
        buffer.append(variables_input)

        if len(buffer) == sequence_length:
            # Definir las columnas de entrada
            input_columns = ['hPV [cm]', 'hSP [cm]', 'PPV [mbar]',
                             'PSP [mbar]', 'Frecuencia de la Bomba [Hz]']

            # Crear el DataFrame
            df = pd.DataFrame(buffer, columns=input_columns)

            # Escalar los datos de entrada
            df['hPV [cm]'] = Escalizado(df,'hPV [cm]',[0,60],[0,1])
            df['hSP [cm]'] = Escalizado(df,'hSP [cm]',[0,60],[0,1])
            df['PPV [mbar]'] = Escalizado(df,'PPV [mbar]',[0,600],[0,1])
            df['PSP [mbar]'] = Escalizado(df,'PSP [mbar]',[0,600],[0,1])
            df['Frecuencia de la Bomba [Hz]'] = Escalizado(df,'Frecuencia de la Bomba [Hz]',[0,60],[0,1])

            X_input = df.to_numpy()
            X_input = X_input.reshape(1, -1, 5)

            # Obtener las variables de salida predichas y desnormalizarlas
            variables_output = np.clip(
                process_variables(X_input)[0]*100, 0, 100)
            
            print(variables_output)
            
            # Guardar las variables de salida para su posterior escritura en el PLC
            aperture_valve_pressure = ua.DataValue(
                ua.Variant(variables_output[0], ua.VariantType.Float))
            aperture_valve_level = ua.DataValue(
                ua.Variant(variables_output[1], ua.VariantType.Float))

            # Obtener los nodos de las variables de salida
            valve_nivel_PLC = client.get_node(
                "ns=2;s=Nivel.PLC Nivel.Valvula Nivel")
            valve_pressure_PLC = client.get_node(
                "ns=2;s=Nivel.PLC Nivel.Valvula Presión")

            # Escribir las variables de salida en el PLC
            await valve_nivel_PLC.set_value(aperture_valve_level)
            await valve_pressure_PLC.set_value(aperture_valve_pressure)

            # Eliminar el elemento más antiguo del buffer (FIFO)
            buffer.pop(0)

    try:
        # Intentar desconectar el cliente OPC UA
        await client.disconnect()
    except Exception as e:
        # En caso de error al desconectar, imprimir el mensaje de excepción y salir de la función
        print(e)
        return

# Ejecutar la función `main()` utilizando `asyncio.run()`
if __name__ == "__main__":
    asyncio.run(main())

