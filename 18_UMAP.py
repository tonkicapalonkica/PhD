# UMAP dimensionality reduction on FDA nonsalts dataset - scaled
import pandas as pd # to deal with .csv files
import umap

# Load your CSV file
df = pd.read_csv("16_FDAnonsaltsRdkit_standardScaled.csv")

# Select only numeric columns
X = df.select_dtypes(include=["number"])

# data already StandardScaled

# Initialize UMAP
reducer = umap.UMAP(n_neighbors=95, min_dist=0.1, n_components=2, random_state=42)

# Fit and transform
embedding = reducer.fit_transform(X)

print(embedding.shape)

#to save in a csv file
# Convert to DataFrame
embedding_df = pd.DataFrame(embedding, columns=['UMAP-1', 'UMAP-2'])

#to save the embdedding along with the original data
combined = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)
combined.to_csv("18_FDA_UMAP_with_original_data.csv", index=False)

# Optional: include labels or other columns from original data
if 'Label' in df.columns:
    embedding_df['Label'] = df['Label']

# Save to CSV
embedding_df.to_csv("18_FDA_UMAPdefault2D.csv", index=False)


# Visualize the results - plot 2D UMAP projection
import matplotlib.pyplot as plt


plt.figure(figsize=(8,6))
plt.scatter(embedding[:, 0], embedding[:, 1], color='#B2182B', s=10, alpha=0.6)
plt.title('UMAP projection of FDA nonsalts dataset')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()

# Visualize the results - plot 2D UMAP projection colored by a specific feature
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=df['MolWt'],       # use molecular weight column for color
    cmap='RdBu',      # color map (you can change this) - 'viridis', 'plasma', 'magma', 'inferno', 'cividis'; 'coolwarm', 'RdBu', 'PiYG'
    s=10,
    alpha=0.6
)
plt.colorbar(label='Molecular Weight (MolWt)')
plt.title('UMAP FDA nonsalts coloured by Molecular Weight')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()

#plot coloured by logP
plt.figure(figsize=(8,6))
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=df['XlogP3'],      # use logP column for colour
    cmap='coolwarm',       # try 'plasma', 'coolwarm', or 'viridis'
    s=10,
    alpha=0.9
)
plt.colorbar(label='XlogP3')
plt.title('UMAP coloured by XlogP3')
plt.xlabel('UMAP-1')
plt.ylabel('UMAP-2')
plt.show()


