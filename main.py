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

X_train, X_temp = train_test_split(X, test_size=0.1, shuffle=False)
X_val, X_test = train_test_split(X_temp, test_size=0.5, shuffle=False)


y_train, y_temp = train_test_split(y, test_size=0.1, shuffle=False)
y_val, y_test = train_test_split(y_temp, test_size=0.5, shuffle=False)

# run pso algorithm
hyperparameters = pso(n_particles=15, n_iterations=30, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# load best pso model from file
pso_model = tf.keras.models.load_model('output_files/pso_model.h5')

# get number of hidden layers and neurons in them
print('Neurons in hidden layers from PSO algorithm: ', hyperparameters)

# evaluate model
results = pso_model.evaluate(X_test, y_test)
pso_loss, pso_accuracy = results
print(f"Loss: {pso_loss}, Accuracy: {pso_accuracy}")
