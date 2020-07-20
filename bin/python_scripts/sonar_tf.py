import numpy as np
import tensorflow as tf
from tensorflow import keras

def prep_sonar_dataset(csv_file):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    #sonar dataset has 60 columns
    _sonar_df = pd.read_csv(csv_file, header=None)  #column 60 is the label
    _sonar_df = _sonar_df.sample(frac=1).reset_index(drop=True)
    ## Split data into training and test
    _X = np.array(_sonar_df[[i for i in range(60)]]).astype(float)
    _y = np.array(_sonar_df[60].apply(lambda x: 1 if x=='M' else 0)).astype(float)
    _X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size=0.2, random_state=18)
    ## Scale data
    s = StandardScaler()
    _X_train = s.fit_transform(_X_train)
    _X_test = s.transform(_X_test)
    return _X_train, _X_test, _y_train, _y_test


def get_performance(_model, _X_test, _y_test):
    ## predict on test data
    _pred = _model.predict(_X_test)
    ## model.predict returns a vector for each input
    _y_hat = [0 if i[0] > i[1] else 1 for i in _pred]
    accuracy = 1-np.sum(np.abs(_y_test - _y_hat))/np.size(_y_test)
    return accuracy


if __name__ == '__main__':
    epochs = 1000
    X_train, X_test, y_train, y_test = prep_sonar_dataset('../../data/sonar_data.csv')

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
    print('validation accuracy:', accuracy)
