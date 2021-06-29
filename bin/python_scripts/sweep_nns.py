from sonar_nn import prep_sonar_dataset, get_performance  #file sonar_tf.py must be in working directory to import
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

epochs = 400
X_train, X_test, y_train, y_test = prep_sonar_dataset('../../data/sonar_data.csv')

## define nn architecture and compile
results = []

for hidden_layer_width in range(3,30,3):
    print('hidden layer width:', hidden_layer_width, end=' : ')
    model = keras.Sequential([
        keras.Input(shape=(60)),
        keras.layers.Dense(hidden_layer_width),
        keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    ## train nn
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    ## get performance
    accuracy = get_performance(model, X_test, y_test)

    ## store results
    print('accuracy: ', accuracy)
    results.append((hidden_layer_width,accuracy))


## create a plot with the results
plt.plot([i[0] for i in results], [i[1] for i in results]) 
plt.xlabel('hidden layer width')
plt.ylabel('accuracy')
plt.savefig('../../output/hidden_layer_width.pdf')
