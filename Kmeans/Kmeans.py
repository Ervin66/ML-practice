import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("College_Data")

print(df.head(),
      df.info(),
      df.describe())

sns.scatterplot(df["Room.Board"], df["Grad.Rate"], hue=df["Private"])
plt.show()


sns.scatterplot("Outstate", "F.Undergrad", data=df, hue="Private")
plt.show()

g = sns.FacetGrid(df, hue="Private")
g.map_dataframe(sns.histplot, x="Outstate")
plt.show()

g = sns.FacetGrid(df, hue="Private")
g.map_dataframe(sns.histplot, x="Grad.Rate")
plt.show()

print(df[df["Grad.Rate"] > 100])

df["Grad.Rate"][df["Grad.Rate"] > 100] = 100

g = sns.FacetGrid(df, hue="Private")
g.map_dataframe(sns.histplot, x="Grad.Rate")
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop(["Private", "Unnamed: 0"], axis=1))

print(kmeans.labels_)
 
df["Cluster"] = df["Private"].apply(lambda x: 1 if x=="Yes" else 0)
print(df["Cluster"])

print(confusion_matrix(df["Cluster"], kmeans.labels_))
print(classification_report(df["Cluster"], kmeans.labels_))