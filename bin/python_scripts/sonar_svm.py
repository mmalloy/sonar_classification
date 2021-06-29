import numpy as np
import pandas as pd
from sklearn import svm


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
    _y_hat = _model.predict(_X_test)
    accuracy = 1-np.sum(np.abs(_y_test - _y_hat))/np.size(_y_test)
    return accuracy


if __name__ == '__main__':
    
    X_train, X_test, y_train, y_test = prep_sonar_dataset('../../data/sonar_data.csv')
    #X_train, X_test, y_train, y_test = prep_sonar_dataset("https://uwmadison.box.com/shared/static/yf9jbcw1espe2djbfw9o1m0bw4ngyyrc.csv")

    ## Train an SVM
    model = svm.SVC()
    model.fit(X_train, y_train)

    ## Predict classes on test data
    #y_test = model.predict(X_test)
    #print('shape y_test', np.shape(y_test))
    ## get performance

    accuracy = get_performance(model, X_test, y_test)
    print('SVM validation accuracy:', accuracy)
