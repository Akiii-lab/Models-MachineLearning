import joblib
import os

def guardar_modelo(modelo, nombre_modelo, carpeta="modelos"):
    # Crear carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    ruta_completa = os.path.join(carpeta, f"{nombre_modelo}.pkl")
    joblib.dump(modelo, ruta_completa)
    print(f" Modelo '{nombre_modelo}' guardado en: {ruta_completa}")

def cargar_modelo(nombre_modelo, carpeta="modelos"):
    ruta_completa = os.path.join(carpeta, f"{nombre_modelo}.pkl")
    if not os.path.exists(ruta_completa):
        raise FileNotFoundError(f" No se encontró el modelo: {ruta_completa}")
    modelo = joblib.load(ruta_completa)
    print(f" Modelo '{nombre_modelo}' cargado desde: {ruta_completa}")
    return modelo