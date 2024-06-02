from tools import preprocess_data
from nn_pso import pso
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# preproess data
preprocess_data('breast-cancer.csv')

# split data
X = pd.read_csv('output_files/X_scaled.csv')
y = pd.read_csv('output_files/y.csv')

X_train, X_temp = train_test_split(X, test_size=0.2, shuffle=False)
X_val, X_test = train_test_split(X_temp, test_size=0.5, shuffle=False)


y_train, y_temp = train_test_split(y, test_size=0.2, shuffle=False)
y_val, y_test = train_test_split(y_temp, test_size=0.5, shuffle=False)

def plot_accuracy_changes(all_accuracies):
    for i, accuracies in enumerate(all_accuracies):
        plt.plot(accuracies, label=f'Model {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.title('Accuracy of all models during PSO iterations')
    plt.savefig('output_files/pso_accuracy_plot.png')
    plt.show()

def plot_val_loss_changes(all_val_losses):
    for i, val_losses in enumerate(all_val_losses):
        plt.plot(val_losses, label=f'Model {i + 1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Validation Loss of all models during PSO iterations')
    plt.savefig('output_files/pso_validation_loss.png')
    plt.show()

# uruchomienie algorytmu pso
hyperparameters, metrics_history, all_accuracies, all_val_losses = pso(n_particles=10, n_iterations=20, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# załadowanie najlepszego modelu z pliku
pso_model = tf.keras.models.load_model('output_files/pso_model.h5')

# wyświetlenie liczby warstw ukrytych i neuronów w nich
print('Neurons in hidden layers from PSO algorithm: ', hyperparameters)

# ocena modelu
results = pso_model.evaluate(X_test, y_test)
pso_loss, pso_accuracy = results
print(f"Loss: {pso_loss}, Accuracy: {pso_accuracy}")

# rysowanie wykresu dokładności
plot_accuracy_changes(all_accuracies)

# rysowanie wykresu straty walidacyjnej
plot_val_loss_changes(all_val_losses)
