import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# pd.options.display.max_columns = None

# TODO
"""
class WeightsHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.weights_ih = []
        self.weights_ho = []

    def on_epoch_end(self, epoch, logs=None):
        weights_ih, _ = self.model.layers[0].get_weights()
        weights_ho, _ = self.model.layers[1].get_weights()
        self.weights_ih.append(weights_ih.flatten())
        self.weights_ho.append(weights_ho.flatten())

class ClassificationErrorHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.classification_errors = []

    def on_train_begin(self, logs=None):
        self.classification_errors = []

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(X) > 0.5
        incorrect_predictions = np.sum(predictions.flatten() != y.flatten())
        error_rate = incorrect_predictions / len(y)
        self.classification_errors.append(error_rate)
"""

def preprocess_data(f_path):
    df = pd.read_csv(f_path)
    df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    df_X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_X_scaled.to_csv('output_files/X_scaled.csv', index=False)
    y.to_csv('output_files/y.csv', index=False)
    print('Data preprocessing done')
