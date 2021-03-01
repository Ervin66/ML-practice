import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel


train_df = pd.read_csv("train.csv")

print(train_df.info(),
      train_df.head(),
      "\n\n\n",
      train_df.describe())


# replace NAN age values with the approate mean per gender and cabin class
print(train_df.groupby(["Sex", "Pclass"]).mean()["Age"])

replace_age = train_df.groupby(["Sex", "Pclass"])["Age"].transform("mean")
train_df["Age"].fillna(replace_age, inplace=True)
print("Undefined values in the 'Age' column: ", train_df["Age"].isna().sum())


train_df["Embarked"].dropna(inplace=True)
train_df.drop(axis=1, inplace=True, columns="Cabin")


train_df["Staff"] = train_df["Fare"].apply(lambda x: 1 if x == 0 else 0)

dummy = pd.get_dummies(train_df[["Sex", "Embarked"]], drop_first=True)

final_df = pd.concat([train_df, dummy], axis=1)

final_df.drop(columns=["Sex", "Embarked", "Name",
                       "PassengerId", "Ticket"], axis=1, inplace=True)

# sns.pairplot(train_df, hue="Survived")
# plt.show()

sns.heatmap(final_df.corr(), annot=True)
plt.show()
y = final_df["Survived"]
X = final_df.drop("Survived", axis=1)


features = X.columns
print(features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


grid = GridSearchCV(LinearSVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_pred = grid.predict(X_test)

for i in range(len(features)):
    print(features[i], grid.best_estimator_.coef_[0][i])
print("\n\n")
print(classification_report(y_test, grid_pred))
print(confusion_matrix(y_test, grid_pred))

test_df = pd.read_csv("test.csv")


replace_age = test_df.groupby(["Sex", "Pclass"])["Age"].transform("mean")
test_df["Age"].fillna(replace_age, inplace=True)




test_df.drop(axis=1, inplace=True, columns="Cabin")


test_df["Staff"] = test_df["Fare"].apply(lambda x: 1 if x == 0 else 0)

dummy = pd.get_dummies(test_df[["Sex", "Embarked"]], drop_first=True)


final_df = pd.concat([test_df, dummy], axis=1)
final_df["Fare"].fillna((final_df["Fare"].mean()), inplace=True)

ids = final_df["PassengerId"].tolist()
final_df.drop(columns=["Sex", "Embarked", "Name",
                       "PassengerId", "Ticket"], axis=1, inplace=True)

print(final_df.info())
X_final = scaler.fit_transform(final_df)

final_pred = grid.predict(X_final)

submission = {"PassengerId": ids, "Survived": final_pred}

submission_df = pd.DataFrame.from_dict(submission)

print(submission_df.head(),
    submission_df.info())

submission_df.to_csv("final.csv", index=False)
