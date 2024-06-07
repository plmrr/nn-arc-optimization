import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import Input # type: ignore
from keras.callbacks import Callback # type: ignore

# pd.options.display.max_columns = None

class MetricsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.accuracy.append(logs.get('accuracy'))
        self.val_loss.append(logs.get('val_loss'))


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
