from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def train_regresion_multivariada(X_train, y_train, X_test, y_test):
    param_grid = {
        'C': [0.01, 0.1, 1, 10], #factor de regularizacion para evitar sobreajuste
        'solver': ['lbfgs', 'saga'], # algoritmo de optimizacion
        'max_iter': [500, 1000], # numero de iteraciones
        'multi_class': ['multinomial'] # para clasificacion multiclase
    }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print("Mejores parámetros encontrados:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("Precisión:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_

def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        'max_depth': [None, 10, 20, 30], # profundidad de arbol
        'min_samples_split': [2, 5, 10], # numero minimo de muestras para dividir un nodo
        'min_samples_leaf': [1, 2, 4] # numero minimo de muestras en una hoja
    }
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return grid.best_estimator_

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200], # numero de arboles
        'max_depth': [None, 10, 20], # profundidad de arbol
        'max_features': ['auto', 'sqrt', 'log2'], # numero de caracteristicas aleatorias
        'min_samples_split': [2, 5, 10] # numero minimo de muestras para dividir un nodo
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return grid.best_estimator_

def train_mlp(X_train, y_train, X_test, y_test):
    param_grid = {
        'hidden_layer_sizes': [(128,), (128, 64)], # dimensiones de las capas ocultas
        'activation': ['relu'], # funcion de activacion
        'alpha': [0.0001, 0.001, 0.01], # factor de regularizacion
        'learning_rate_init': [0.001, 0.01], # tasa de aprendizaje
        'max_iter': [300, 500] # numero de iteraciones
    }
    grid = GridSearchCV(MLPClassifier(), param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_