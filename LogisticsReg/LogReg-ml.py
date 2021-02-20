import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv("advertising.csv")

# data exploration
print(df.head(),
    df.info(),
    df.describe())

df["Age"].plot.hist()
plt.show()

sns.jointplot(x="Age",
    y="Area Income",
    data=df)
plt.show()

sns.jointplot(x="Age",
    y="Daily Time Spent on Site",
    kind="kde",
    data=df)
plt.show()

sns.jointplot(x="Daily Time Spent on Site",
    y="Daily Internet Usage",
    kind="kde",
    data=df)
plt.show()

sns.pairplot(df, hue="Clicked on Ad")
plt.show()


# model training
X = df[["Age", "Daily Internet Usage", "Area Income", "Male"]]
y = df["Clicked on Ad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predict = logmodel.predict(X_test)


print(classification_report(y_test, predict))