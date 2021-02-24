import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


cancer = load_breast_cancer()
print(cancer["DESCR"])

df = pd.DataFrame(cancer["data"], columns=cancer["feature_names"])

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer["target"])
plt.xlabel("First PCA")
plt.ylabel("Second PCA")
plt.show()

df_comp = pd.DataFrame(pca.components_, columns=cancer["feature_names"])

sns.heatmap(df_comp)
plt.show()
