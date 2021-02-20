import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#initial inspection
df = pd.read_csv("Ecommerce Customers.csv")
print(df.head(),
      df.info(),
      df.describe())

sns.jointplot(data=df,
              x="Time on Website",
              y="Yearly Amount Spent")
plt.show()

sns.jointplot(data=df,
              x="Time on App",
              y="Yearly Amount Spent")
plt.show()

sns.jointplot(data=df,
              x="Time on App",
              y="Length of Membership",
              kind="hex")
plt.show()

sns.pairplot(df)
plt.show()

sns.lmplot(data=df,
           x="Length of Membership",
           y="Yearly Amount Spent")
plt.show()


#Linear Regression
X=df[["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]]

y=df["Yearly Amount Spent"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.coef_)

predict = lm.predict(X_test)

sns.scatterplot(y_test,
                predict)
plt.show()


#Performance evaluation
mae = metrics.mean_absolute_error(y_test, predict)
mse = metrics.mean_squared_error(y_test, predict)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predict))

sns.distplot((y_test - predict))
sns.show()
