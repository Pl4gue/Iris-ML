from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris_dataset=load_iris()

X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

knn= KNeighborsClassifier(n_neighbors=1)
print(knn.fit(X_train,y_train))

X_new= np.array([[5,2.9,1,0.2]])

y_predict=knn.predict(X_test)
print(y_predict) #predicted iris categories for test values in X_test

print("accuracy: {:.2f}".format(np.mean(y_predict==y_test))+
	" | {:.2f}".format(knn.score(X_test,y_test)))