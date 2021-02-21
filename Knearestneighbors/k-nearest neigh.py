import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("KNN_Project_Data")

print(df.head(),
      df.describe(),
      df.info())

sns.pairplot(data=df, hue="TARGET CLASS")
plt.show()
scaler = StandardScaler()

scaler.fit(df.drop("TARGET CLASS", axis=1))
scaled_features = scaler.transform(df.drop("TARGET CLASS", axis=1))


df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())


X = df_feat
y = df["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
error_m = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_m.append(np.mean(y_test != pred))

plt.plot(range(1, 40), error_m, color="blue", linestyle="--", marker="o")
plt.title("Error Rate vs K value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()

knn = KNeighborsClassifier(n_neighbors=31)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
 