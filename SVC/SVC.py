import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


iris = sns.load_dataset('iris')
print(type(iris))
sns.pairplot(iris, hue="species")
plt.show()

setosa = iris[iris["species"] == "satosa"]
sns.kdeplot(setosa["sepal_length"],
            setosa["sepal_width"])
plt.show()

keys = ["sepal_width", "sepal_length", "petal_length", "petal_width"]
X = iris.drop("species", axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)

model = SVC()

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


param_grid = {"C": [0.1, 1, 10, 100, 100],
              "gamma": [0.0001, 0.001, 0.01, 0.1, 1]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_pred = grid.predict(X_test)

print(classification_report(y_test, grid_pred))
print(confusion_matrix(y_test, grid_pred))
