# -*- coding: utf-8 -*-

# Importación de librerías
import tensorflow as tf
import asyncio
import logging
import h5py
from asyncua import Client
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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
ruta_modelo = 'Modelo_Nivel_Presion.h5'

# Cargar el modelo entrenado
with h5py.File(ruta_modelo, 'r') as file:
    model_cargado = tf.keras.models.load_model(file)


# Crea una función para procesar las variables de entrada y obtener las variables de salida predichas
def process_variables(variables_input):
    # Crear una instancia de MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizar las variables de entrada
    variables_input_normalized = scaler.fit_transform(variables_input)

    # Transformar los datos en tensores
    variables_input_tensor = tf.convert_to_tensor(
        variables_input_normalized, dtype=tf.float32)

    # Obtener las variables de salida predichas
    variables_output_pred = model_cargado.predict(variables_input_tensor)

    # Desnormalizar las variables de salida predichas
    variables_output = scaler.inverse_transform(variables_output_pred)

    return variables_output


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

    # Variables de Entrada al modelo entregadas por el PLC
    nivel_sp = await client.get_node("ns=2;s=Nivel.PLC Nivel.SP Nivel").get_value()
    nivel_pv = await client.get_node("ns=2;s=Nivel.PLC Nivel.Nivel").get_value()
    presion_sp = await client.get_node(
        "ns=2;s=Nivel.PLC Nivel.SP Presión").get_value()
    presion_pv = await client.get_node("ns=2;s=Nivel.PLC Nivel.Presión").get_value()
    frec_bomba = await client.get_node(
        "ns=2;s=Nivel.PLC Nivel.Frec Bomba").get_value()

    variables_input = np.array(
        [[nivel_pv, nivel_sp, presion_pv, presion_sp, frec_bomba]])
    print(variables_input)

    # Buffer FIFO Sliding Window buffer para almacenar las variables de entrada
    # While infinito para capturar las variables de entrada
    # Capturar el CTRL C para detener el programa

    variables_output = process_variables(variables_input)
    print(variables_output)

    # Agregar una pausa de 2 segundos para permitir que se realicen las operaciones.
    await asyncio.sleep(2)
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


# ----------------------- Código para revisar la supscripción ------------------------- -------------------- a los cambios de las variables de entrada del modelo ----------#

""" 
def subscribe_opc_variables():
    # Obtener los nodos de las variables de entrada y salida
    # Reemplaza con los NodeIds de tus variables de entrada
    node_input1 = client.get_node("ns=2;s=Entrada1")
    node_input2 = client.get_node("ns=2;s=Entrada2")
    node_input3 = client.get_node("ns=2;s=Entrada3")
    node_input4 = client.get_node("ns=2;s=Entrada4")
    node_input5 = client.get_node("ns=2;s=Entrada5")

    # Reemplaza con los NodeIds de tus variables de salida
    node_output1 = client.get_node("ns=2;s=Salida1")
    node_output2 = client.get_node("ns=2;s=Salida2")

    # Suscribirse a los cambios de las variables de entrada
    handle1 = node_input1.subscribe_data_change(data_change_callback)
    handle2 = node_input2.subscribe_data_change(data_change_callback)
    handle3 = node_input3.subscribe_data_change(data_change_callback)
    handle4 = node_input4.subscribe_data_change(data_change_callback)
    handle5 = node_input5.subscribe_data_change(data_change_callback)

    # Función callback para procesar los datos cuando cambian las variables de entrada
    def data_change_callback(node, data):
        # Obtener los valores de las variables de entrada
        variables_input1 = np.array(data.monitored_items[0].value.Value)
        variables_input2 = np.array(data.monitored_items[1].value.Value)
        variables_input3 = np.array(data.monitored_items[2].value.Value)
        variables_input4 = np.array(data.monitored_items[3].value.Value)
        variables_input5 = np.array(data.monitored_items[4].value.Value)

        # Combinar las variables de entrada en una matriz
        variables_input = np.column_stack(
            (variables_input1, variables_input2, variables_input3, variables_input4, variables_input5))

        # Procesar las variables de entrada y obtener las variables de salida predichas
        variables_output = process_variables(variables_input)

        # Separar las variables de salida
        variables_output1 = variables_output[:, 0]
        variables_output2 = variables_output[:, 1]

        # Escribir los valores de las variables de salida en los nodos correspondientes
        node_output1.set_value(variables_output1)
        node_output2.set_value(variables_output2)

    # Mantener la ejecución en espera para recibir las actualizaciones del OPC
    client.run_subscriptions()


subscribe_opc_variables() """

# ----------   ----- Código para escribir en el OPC ---------------------------------#

""" # Variables de Salida del modelo entregadas al PLC
valvula_nivel = client.get_node("ns=2;s=Nivel.PLC Nivel.Valvula Nivel")
valvula_presion = client.get_node(
    "ns=2;s=Nivel.PLC Nivel.Valvula Presión") """

""" prueba_set = client.get_node("ns=2;s=Nivel.PLC Nivel.Valvula Nivel")

valueWrite = ua.DataValue(ua.Variant(60, ua.VariantType.Float))
await prueba_set.set_value(valueWrite)

value = await prueba_set.get_value()
print("Value: ", value) """
