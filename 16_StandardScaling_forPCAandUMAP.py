import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a DataFrame
input_file = "15_FDAnonsaltsRdkit_variationNoNan.csv"  # input file name
output_file = "16_FDAnonsaltsRdkit_standardScaled.csv"  # output file name

# Read the CSV file
df = pd.read_csv(input_file)

# Initialize the StandardScaler - to scale all the columns in a .csv file to have a mean of 0 and a standard deviation of 1.
# standardizes features by removing the mean and scaling to unit variance (mean = 0, standard deviation = 1). 
scaler = StandardScaler()

# Apply scaling only to numeric columns (non-numeric columns are unchanged)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns]) #Fits the scaler to the numeric columns and transforms them to the standardized scale.

# Save the scaled DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Scaled data saved to {output_file}")