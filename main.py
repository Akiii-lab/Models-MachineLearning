from src.data_loader import data_loader_from_folder
from src import models
from src.evaluate import evaluar_modelo
from src.comparate import comparar_modelos
from src import save_model 
from src import visual

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar imágenes
X, y = data_loader_from_folder("archive/", size=(32, 32))

# Dividir y escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar modelos
modelo_log = models.train_regresion_multivariada(X_train, y_train)
modelo_tree = models.train_decision_tree(X_train, y_train)
modelo_rf = models.train_random_forest(X_train, y_train)
modelo_mlp = models.train_mlp(X_train, y_train)


# Evaluar modelos
evaluar_modelo(modelo_log, X_test, y_test, "Regresión Logística Multinomial")
evaluar_modelo(modelo_tree, X_test, y_test, "Árbol de Decisión")
evaluar_modelo(modelo_rf, X_test, y_test, "Random Forest")
evaluar_modelo(modelo_mlp, X_test, y_test, "Perceptrón Multicapa")

# Diccionario con los modelos entrenados
modelos_entrenados = {
    "Regresión Multivariada": modelo_log,
    "Árbol de Decisión": modelo_tree,
    "Random Forest": modelo_rf,
    "Perceptrón Multicapa": modelo_mlp
}

# Comparar todos los modelos
df_resultados = comparar_modelos(modelos_entrenados, X_test, y_test)

#grafica de comparacion
graficar_comparacion(df_resultados)


# esto que esta aca es como el te de manzanilla no hace nada pero cae bien
# para guardar modelos si es necesario
# guardar_modelo(modelo_log, "logistic_regression")
# guardar_modelo(modelo_tree, "decision_tree")
# guardar_modelo(modelo_rf, "random_forest")
# guardar_modelo(modelo_mlp, "mlp")

# Guardar scaler
# guardar_modelo(scaler, "scaler_digitos")

# para cargar modelos
# modelo_cargado = cargar_modelo("random_forest")
# y_pred = modelo_cargado.predict(X_test)

# Cargar scaler
# scaler = cargar_modelo("scaler_digitos")


#falta hacer una funcion que haga una prediccion pero no se si es obligatoria

# Ejemplo para MLP
y_pred = modelo_mlp.predict(X_test)
plot_confusion_matrix(y_test, y_pred, title="Matriz de Confusión - MLP")
show_classification_report(y_test, y_pred)

# Mostrar iamgenes
plot_sample_images(X_train, y_train, n=10, img_shape=(32, 32))
