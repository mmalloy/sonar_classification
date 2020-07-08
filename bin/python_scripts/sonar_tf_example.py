import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

sonar_df = pd.read_csv('sonar_data.csv', header=None)  #column 60 is the label
sonar_df = sonar_df.sample(frac=1).reset_index(drop=True)
print(sonar_df.dtypes)

## Split data into training and test
X = np.array(sonar_df[[i for i in range(60)]])
y = np.array(sonar_df[[60]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

## Scale data
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

## Train an SVM

## Setup
model = keras.Sequential([
    keras.Input(shape=(60)),
    keras.layers.Dense(15),
    keras.layers.Dense(15),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## train model
pred = model.fit(X_train, y_train, epochs=5000)

## predict on test data
pred = model.predict(X_test)

## model.predict returns a vector for each input
y_hat = [0 if i[0] > i[1] else 1 for i in pred]

print(classification_report(y_test, y_hat))
