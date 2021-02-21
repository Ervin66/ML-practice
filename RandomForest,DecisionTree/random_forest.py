import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("loan_data.csv")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(df.head(),
      df.describe(),
      df.info())

print(df["credit.policy"][df["credit.policy"] == 0])

sns.distplot(df["fico"][df["credit.policy"] == 0],
             color="blue", label="credit=0")
sns.distplot(df["fico"][df["credit.policy"] == 1],
             color="red", label="credit=1")
plt.legend()
plt.show()

sns.distplot(df["fico"][df["not.fully.paid"] == 0],
             color="blue", label="credit=0")
sns.distplot(df["fico"][df["not.fully.paid"] == 1],
             color="red", label="credit=1")
plt.legend()
plt.show()

sns.countplot(df["purpose"], hue=df["not.fully.paid"])
plt.show()

sns.jointplot(data=df,
              x="int.rate",
              y="fico"
              )
plt.show()

sns.lmplot(x="fico",
           y="int.rate",
           data=df,
           hue="credit.policy",
           col="not.fully.paid")
plt.show()

final_df = pd.get_dummies(df, columns=["purpose"], drop_first=True)

X = final_df.drop("not.fully.paid", axis=1)
y = final_df["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

prediction = dtree.predict(X_test)

print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))

rfm = RandomForestClassifier()
rfm.fit(X_train, y_train)
rfm_pred = rfm.predict(X_test)


print(confusion_matrix(y_test, rfm_pred))
print(classification_report(y_test, rfm_pred))