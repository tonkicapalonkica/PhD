import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
input_file = "16_FDAnonsaltsRdkit_standardScaled.csv"  # my dataset file
df = pd.read_csv(input_file)

# Select only numeric columns for clustering
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
data = df[numeric_columns]

# Perform K-Means clustering
n_clusters = 20  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(data)

# Save the clustered data to a new CSV file
output_file = "19_FDAnonsaltsK-means.csv"
df.to_csv(output_file, index=False)
print(f"Clustering completed. Results saved to '{output_file}'.")

# Visualize the clusters (if the data is 2D or reduced to 2D)
if data.shape[1] == 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=df['Cluster'], cmap='tab20', s=10, alpha=0.7)
    plt.title("K-Means Clustering FDA nonsalts (20 Clusters)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.colorbar(label="Cluster")
    plt.show()
else:
    print("Visualization skipped: Data has more than 2 dimensions.")