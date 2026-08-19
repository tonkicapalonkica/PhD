import pandas as pd

# Load the CSV file into a DataFrame
input_file = "04_APIs_moldescr_Rdkit_withnames.csv"  # input file name
output_file = "06_APIs_Rdkit_withnames_variation.csv"  # output file name

# Read the CSV file
df = pd.read_csv(input_file)

# Remove columns where all values are the same
df_cleaned = df.loc[:, (df.nunique() > 1)]

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv(output_file, index=False)

print(f"Columns with unique values saved to {output_file}")