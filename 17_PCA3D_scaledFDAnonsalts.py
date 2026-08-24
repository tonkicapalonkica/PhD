import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to enable 3D plotting

plt.rcParams["font.family"] = "Calibri" #sets the font used for all text in the plot
plt.rcParams["text.color"] = "black" #makes the title fully black
plt.rcParams["axes.labelcolor"] = "black" #makes the axis labels fully black
plt.rcParams["xtick.color"] = "black" #makes the x axis tick numbers fully black
plt.rcParams["ytick.color"] = "black" #makes the y axis tick numbers fully black

# Load the dataset
input_file = "16_FDAnonsaltsRdkit_standardScaled.csv"  # already scaled dataset file
df = pd.read_csv(input_file)

# Select only numeric columns for PCA (non-numeric columns are ignored, i.e. "SMILES", "NAME", etc.)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
data = df[numeric_columns]

# Perform PCA
n_components = 3  # Number of principal components to keep (3D)
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data)

# Create a DataFrame for the principal components
pca_columns = [f"PC{i+1}" for i in range(n_components)]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

# Add the principal components to the original DataFrame
result_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

# Save the result to a new CSV file
result_df.to_csv("17_PCA3D_scaledFDAnonsalts_thesis.csv", index=False)

# Report how much variance each principal component explains
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance_ratio)
print("Cumulative variance explained:", explained_variance_ratio.cumsum())
variance_df = pd.DataFrame({
    "Component": pca_columns,
    "ExplainedVarianceRatio": explained_variance_ratio,
    "CumulativeVarianceExplained": explained_variance_ratio.cumsum()
})
variance_df.to_csv("17_PCA3D_scaledFDAnonsalts_explainedVariance.csv", index=False)

# Report which descriptors contribute most to each principal component
loadings_df = pd.DataFrame(
    pca.components_.T,      # transpose so rows = descriptors, columns = PCs
    index=numeric_columns,  # descriptor names
    columns=pca_columns
)
loadings_df.to_csv("17_PCA3D_scaledFDAnonsalts_loadings.csv")

# Plot the first three principal components (if applicable)
if n_components >= 3:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d') #create the 3D axes first, before setting the title, so no stray 2D axes gets created behind it
    ax.set_title("b) 3D PCA", fontsize=18, pad=1, loc="left") #loc="left" aligns the title to the left edge of the plot

    # Make the 3D panes (walls) white instead of the default grey
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')

    # Plot 3D scatter
    ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.7, color='#B22222')  # Firebrick red colour
    # Labels
    ax.set_xlabel('PC1', fontsize=16, labelpad=5)  # labelpad adds a small gap between the label and the tick numbers
    ax.set_ylabel('PC2', fontsize=16, labelpad=5)
    ax.set_zlabel('PC3', fontsize=16, labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=14)  # Set tick label size for x and y axes
    ax.tick_params(axis='z', which='major', labelsize=14)  # Set tick label size for z axis

    #axis scale settings - change these numbers to control the range shown on each axis
    x_min, x_max = -15, 20 #set to None, None to let matplotlib choose automatically
    y_min, y_max = -5, 15
    z_min, z_max = -20, 5

    ax.set_xlim(x_min, x_max) #set_xlim/ylim/zlim ignore a bound left as None, so no extra checks are needed
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    plt.grid(False) #turn off the grid
    plt.show()

print("PCA completed. Results saved to '17_PCA3D_scaledFDAnonsalts_thesis.csv'.")