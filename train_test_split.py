from sklearn.datasets import load_iris
iris_dataset=load_iris()

import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("dimensions of X_train: {}".format(X_train.shape)) 
#75% of dataset for training (x values, data (petal,sepal)) 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
print("dimensions of y_train: {}".format(y_train.shape))
 #75% of dataset for training (y values, target values 0/1/2 'setosa' 'versicolor' 'virginica')

print("dimensions of X_test: {}".format(X_test.shape)) #25% of dataset for testing (x values)
print("dimensions of y_test: {}".format(y_test.shape)) #25% of dataset for testing (y values)

