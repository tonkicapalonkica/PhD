import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
input_file = "16_FDAnonsaltsRdkit_standardScaled.csv"  # already scaled dataset file
df = pd.read_csv(input_file)

# Select only numeric columns for PCA (non-numeric columns are ignored, i.e. "SMILES", "NAME", etc.)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
data = df[numeric_columns]

# Perform PCA
n_components = 2  # Number of principal components to keep (2D)
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data)

# Create a DataFrame for the principal components
pca_columns = [f"PC{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

# Add the principal components to the original DataFrame
result_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

# Save the result to a new CSV file
result_df.to_csv("17_PCA2D_scaledFDAnonsalts.csv", index=False)

# Plot the first two principal components (if applicable)
if n_components >= 2:
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7, color='#B22222')  # Firebrick red colour
    plt.title("PCA2D: First Two Principal Components")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(False) #turn off the grid
    plt.show()

print("PCA completed. Results saved to '17_PCA2D_scaledFDAnonsalts.csv'.")