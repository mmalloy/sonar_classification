import sonar_tf #file sonar_tf.py must be in working directory to import
import numpy as np
import tensorflow as tf
from tensorflow import keras

epochs = 1000
X_train, X_test, y_train, y_test = prep_sonar_dataset('../../data/sonar_data.csv')

## define nn architecture and compile
results = []
for i range(100):
    model = keras.Sequential([
        keras.Input(shape=(60)),
        keras.layers.Dense(i),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    ## train nn
    model.fit(X_train, y_train, epochs=epochs)

    ## get performance
    accuracy = get_performance(model, X_test, y_test)

    ## store results
    results.append((i,accuracy))
