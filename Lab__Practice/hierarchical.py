import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Load USArrests dataset
# from ISLP import load_data
# USArrests = load_data('USArrests')
# USArrests.index.name = 'State'

# Load the dataset directly from GitHub
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USArrests.csv'
USArrests = pd.read_csv(url, index_col=0)
print(USArrests.head())


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(USArrests)

# Perform hierarchical clustering using complete linkage and Euclidean distance
linked = linkage(scaled_data, method='complete', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=USArrests.index.to_list(), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage, Euclidean Distance)')
plt.xlabel('States')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


# Cut the dendrogram to form 3 clusters
clusters = fcluster(linked, t=3, criterion='maxclust')

# Create a DataFrame with cluster assignments
cluster_df = pd.DataFrame({'State': USArrests.index, 'Cluster': clusters})
print(cluster_df.sort_values('Cluster'))


# Perform hierarchical clustering using correlation distance
# 1 - correlation coefficient gives a distance metric
from scipy.spatial.distance import pdist, squareform

corr_distance = pdist(scaled_data, metric='correlation')
linked_corr = linkage(corr_distance, method='complete')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked_corr, labels=USArrests.index.to_list(), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage, Correlation Distance)')
plt.xlabel('States')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


# Cut dendrogram to form 3 clusters
clusters_corr = fcluster(linked_corr, t=3, criterion='maxclust')

# Create cluster assignments for correlation distance
cluster_corr_df = pd.DataFrame({'State': USArrests.index, 'Cluster': clusters_corr})
print(cluster_corr_df.sort_values('Cluster'))

