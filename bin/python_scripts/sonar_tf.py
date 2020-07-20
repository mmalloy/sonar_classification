import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

def prep_sonar_dataset(csv_file):
    _sonar_df = pd.read_csv(csv_file, header=None)  #column 60 is the label
    _sonar_df = _sonar_df.sample(frac=1).reset_index(drop=True)

    ## Split data into training and test
    _X = np.array(_sonar_df[[i for i in range(60)]])
    _y = np.array(_sonar_df[[60]])
    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state=18)

    ## Scale data
    s = StandardScaler()
    _X_train = s.fit_transform(_X_train)
    _X_test = s.transform(_X_test)
    return _X_train, _X_test, _y_train, _y_test


def get_peforamance(_model):
    ## predict on test data
    pred = model.predict(X_test)
    ## model.predict returns a vector for each input
    y_hat = [0 if i[0] > i[1] else 1 for i in pred]
    print(classification_report(y_test, y_hat))
    print('accuracy: ', np.sum(np.abs(y_test-y_hat)))
    return np.sum(np.abs(y_test-y_hat))

if __name__ == '__main__':
    epochs = 5000
    X_train, X_test, y_train, y_test = prep_sonar_dataset('sonar_data.csv')

    ## define nn architecture and compile
    model = keras.Sequential([
        keras.Input(shape=(60)),
        keras.layers.Dense(15),
        keras.layers.Dense(15),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    ## train nn
    model.fit(X_train, y_train, epochs=epochs)

    ## get performance
    accuracy = get_performance(model, X_test, y_test)
    print('final peformance:', accuracy)
