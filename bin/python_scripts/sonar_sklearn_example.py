import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

#sonar_df = pd.read_csv('../../data/sonar_data.csv', header=None)  #column 60 is the label
sonar_df = pd.read_csv("https://uwmadison.box.com/shared/static/yf9jbcw1espe2djbfw9o1m0bw4ngyyrc.csv", header=None)  #column 60 is the label

sonar_df = sonar_df.sample(frac=1).reset_index(drop=True)

## Split data into training and test
X = np.array(sonar_df[[i for i in range(60)]])
y = np.array(sonar_df[[60]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

## Scale data
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

## Train an SVM
sc = svm.SVC()
sc.fit(X_train, y_train)

## Predict classes on test data
y_svm = sc.predict(X_test)
print(classification_report(y_test, y_svm))
