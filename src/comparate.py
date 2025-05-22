from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# tabla de comparación
def comparar_modelos(modelos_dict, X_test, y_test):
    resultados = []

    for nombre, modelo in modelos_dict.items():
        y_pred = modelo.predict(X_test)
        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision (macro)": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "Recall (macro)": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "F1-score (macro)": f1_score(y_test, y_pred, average='macro', zero_division=0)
        })

    df_resultados = pd.DataFrame(resultados)
    df_ordenado = df_resultados.sort_values(by="F1-score (macro)", ascending=False).reset_index(drop=True)
    print("\nComparación de modelos:")
    print(df_ordenado)

    return df_ordenado

def graficar_comparacion(df_resultados):
    plt.figure(figsize=(12, 6))
    ancho_barras = 0.35
    x = range(len(df_resultados))

    # Barras para Accuracy
    plt.bar(x, df_resultados["Accuracy"], width=ancho_barras, label="Accuracy", align='center')

    # Barras para F1-score
    plt.bar([i + ancho_barras for i in x], df_resultados["F1-score (macro)"], width=ancho_barras, label="F1-score", align='center')

    # Etiquetas de modelos
    nombres_modelos = df_resultados["Modelo"]
    plt.xticks([i + ancho_barras / 2 for i in x], nombres_modelos, rotation=45)

    # Títulos y leyenda
    plt.title("Comparación de Modelos - Accuracy vs F1-score")
    plt.ylabel("Valor")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    # Mostrar gráfico
    plt.show()
