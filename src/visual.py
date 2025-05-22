import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Muestra la matriz de confusión.
def plot_confusion_matrix(y_true, y_pred, labels=None, title="Matriz de Confusión"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Muestra una fila de imágenes aleatorias del conjunto X con sus etiquetas y.
def plot_sample_images(X, y, n=10, img_shape=(32, 32)):
    plt.figure(figsize=(n, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(img_shape), cmap='gray')
        plt.title(str(y[i]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Imprime un resumen del desempeño del modelo por clase.
def show_classification_report(y_true, y_pred, target_names=None):
    print("\n Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))