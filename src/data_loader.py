import os
import numpy as np
from PIL import Image

# objetivo del codigo es poder cargar las imagenes a un matriz de 1D
def data_loader_from_folder(data_dir, size=(32, 32)):
    X = []
    y = []
    for etiqueta in os.listdir(data_dir):
        clase_path = os.path.join(data_dir, etiqueta)
        if os.path.isdir(clase_path):
            for archivo in os.listdir(clase_path):
                img_path = os.path.join(clase_path, archivo)
                try:
                    img = Image.open(img_path).convert("L").resize(size)
                    X.append(np.array(img).flatten())
                    y.append(int(etiqueta))
                except:
                    print(f"Error con la imagen: {img_path}")
    return np.array(X), np.array(y)