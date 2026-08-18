from rdkit import Chem #imports Chem module from rdkit - allows to calculate molecular descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors # also needed to calculate molecular descriptors
from rdkit.Chem import Descriptors # contains a list of predefined molecular descriptors available in RDKit
import csv # imports csv module to read from and write to CSV files (built-in Python library)

def calculate_descriptors(smiles_list): # defines a function named calculate_descriptors that takes a list of SMILES strings as input
    # Create a list of descriptor names
    descriptor_names = [desc[0] for desc in Descriptors._descList] # I don't understand this.
    
    # Initialize the descriptor calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    # creates an object called moleculardescriptorcalculator that can calculate all the descriptors listed in descriptor_names for a given molecule
    
    # Prepare a list to store results
    results = []
    # initialises an empty list called results to store the calculated descriptor values for each molecule
    
    for smiles in smiles_list: #this is a loop that iterates over each SMILES string in the input list
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles) # converts the SMILES string to an RDKit molecule object using the molfromsmiles function from the chem module
        if mol is None: # checks if the conversion was successful (i.e., if mol is not None and smiles string is valid)
            print(f"Invalid SMILES: {smiles}") # if smiles is invalid, it prints a message indicating that the SMILES string is invalid
            results.append(None) # appends None to the results list to indicate that no descriptors could be calculated for this invalid SMILES string
            continue #skips the rest of the loop for the invalid SMILES and moves to the next SMILES string in the list
        
        # Calculate descriptors
        descriptors = calculator.CalcDescriptors(mol) #calculates all the descriptors for the molecule object (smiles string converted to mol?)
        results.append(dict(zip(descriptor_names, descriptors))) 
        #combines the descriptor names and their corresponding values into a dictionary using the zip function, and appends this dictionary to the results list    
    return descriptor_names, results

# Read SMILES strings from the CSV file
def read_smiles_from_csv(file_path):
    smiles_list = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row:  # Ensure the row is not empty
                    smiles_list.append(row[0])  # Assuming SMILES are in the first column
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    return smiles_list

# Write descriptors to a CSV file
def write_descriptors_to_csv(output_file, smiles_list, descriptor_names, descriptors):
    try:
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header row
            header = ["SMILES"] + descriptor_names
            writer.writerow(header)
            
            # Write the descriptor values
            for i, smiles in enumerate(smiles_list):
                if descriptors[i] is not None:
                    row = [smiles] + list(descriptors[i].values())
                    writer.writerow(row)
                else:
                    writer.writerow([smiles] + ["Invalid SMILES"] * len(descriptor_names))
        
        print(f"Descriptors written to '{output_file}' successfully.")
    except Exception as e:
        print(f"Error writing to file: {e}")

# Example usage
file_path = "03 FDA_smiles.csv"  # Path to your SMILES file
output_file = "04 FDA_Rdkit_forMw.csv"  # Output file for descriptors

smiles_list = read_smiles_from_csv(file_path)
if smiles_list:
    descriptor_names, descriptors = calculate_descriptors(smiles_list)
    write_descriptors_to_csv(output_file, smiles_list, descriptor_names, descriptors)