# src/ml/models/shared/directional_prediction.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def calculate_directions(data):
    """
    Calcula las direcciones de cambio en los datos de entrada (Open, Close).
    """
    directions = np.diff(data, axis=0)  # Diferencias a lo largo del tiempo
    directional_data = np.zeros((directions.shape[0], 2, 2))  # Matriz (n_samples-1, 2 features, 2 clases)

    for i in range(directions.shape[0]):
        for j in range(2):  # Para cada feature (Open y Close)
            if directions[i, j] > 0:
                directional_data[i, j] = [0, 1]  # Up
            else:
                directional_data[i, j] = [1, 0]  # Down

    return directional_data

def calculate_movement_metrics(true_labels, predicted_labels, model_name, set_type, data_type):
    """
    Calcula métricas de clasificación (accuracy, recall, precision, F1-score) 
    y muestra la matriz de confusión si es el set de test.
    """
    metrics = {}

    # Calcular métricas para cada feature por separado
    for i, feature in enumerate(["Open", "Close"]):
        acc = accuracy_score(true_labels[:, i], predicted_labels[:, i]) * 100
        metrics[f'{set_type} Accuracy ({feature})'] = acc

        if set_type == "Test":
            recall = recall_score(true_labels[:, i], predicted_labels[:, i], pos_label=1) * 100
            precision_rise = precision_score(true_labels[:, i], predicted_labels[:, i], pos_label=1) * 100
            precision_fall = precision_score(true_labels[:, i], predicted_labels[:, i], pos_label=0) * 100
            f1 = f1_score(true_labels[:, i], predicted_labels[:, i], pos_label=1) * 100


            print(f'{model_name} ({data_type}) | Test Accuracy: {acc:.2f}%')
            print(f'  - Recall ({feature}): {recall:.2f}%')
            print(f'  - Precision (Rise) ({feature}): {precision_rise:.2f}%')
            print(f'  - Precision (Fall) ({feature}): {precision_fall:.2f}%')
            print(f'  - F1 Score ({feature}): {f1:.2f}%')

            metrics.update({
                f'Recall ({feature})': recall,
                f'Precision (Rise) ({feature})': precision_rise,
                f'Precision (Fall) ({feature})': precision_fall,
                f'F1 Score ({feature})': f1
            })

            # Matriz de confusión
            cm = confusion_matrix(true_labels[:, i], predicted_labels[:, i])
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], cbar=False)

            # Agregar porcentajes
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    plt.text(c + 0.5, r + 0.55, f'\n({cm_norm[r, c]:.2%})',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='black',
                             fontsize=9)

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_name} Confusion Matrix ({data_type} Data - {feature})')
            plt.show()

    return metrics

def evaluate_directional_movement(actual_train, pred_train, actual_val, pred_val, actual_test, pred_test, model_name, data_type, using_darts=True):
    """
    Evalúa la predicción de direcciones de movimiento para cada feature (Open y Close).
    """
    # Extraer valores de los diccionarios si se usa Darts
    if using_darts:
        train_y = actual_train.values()
        train_predictions = pred_train.values()
        val_y = actual_val.values()
        val_predictions = pred_val.values()
        test_y = actual_test.values()
        test_predictions = pred_test.values()
    else:
        train_y, train_predictions = actual_train, pred_train
        val_y, val_predictions = actual_val, pred_val
        test_y, test_predictions = actual_test, pred_test

    # Calcular direcciones para cada conjunto
    true_directions_train = calculate_directions(train_y)
    pred_directions_train = calculate_directions(train_predictions)
    true_directions_val = calculate_directions(val_y)
    pred_directions_val = calculate_directions(val_predictions)
    true_directions_test = calculate_directions(test_y)
    pred_directions_test = calculate_directions(test_predictions)

    # Convertir a etiquetas (0=Down, 1=Up) para cada feature
    true_labels_train = np.argmax(true_directions_train, axis=2)
    predicted_labels_train = np.argmax(pred_directions_train, axis=2)
    true_labels_val = np.argmax(true_directions_val, axis=2)
    predicted_labels_val = np.argmax(pred_directions_val, axis=2)
    true_labels_test = np.argmax(true_directions_test, axis=2)
    predicted_labels_test = np.argmax(pred_directions_test, axis=2)

    # Calcular métricas
    metrics_train = calculate_movement_metrics(true_labels_train, predicted_labels_train, model_name, "Train", data_type)
    metrics_val = calculate_movement_metrics(true_labels_val, predicted_labels_val, model_name, "Val", data_type)
    metrics_test = calculate_movement_metrics(true_labels_test, predicted_labels_test, model_name, "Test", data_type)

    # Unificar métricas de train y val con test
    metrics_test.update(metrics_val)
    metrics_test.update(metrics_train)

    return true_labels_test, predicted_labels_test, metrics_test
