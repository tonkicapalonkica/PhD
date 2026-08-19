import csv

from rdkit import Chem

def peg_ppg_peg_smiles(x, y, z):
    """
    Generate SMILES for HO-(EO)x-(PO)y-(EO)z-H
    EO = -CH2-CH2-O-
    PO = -CH(CH3)-CH2-O-
    """
    # Start with terminal OH (O with implicit H)
    smi = "O"

    # First PEG block: x EO units
    # Each EO: CCO (O already present from previous step)
    smi += "CCO" * x

    # PPG block: y PO units
    # Each PO: C(C)CO  (CH(CH3)-CH2-O)
    # We already have an O from previous segment, so we add C(C)CO per unit
    po_unit = "C(C)CO"
    smi += po_unit * y

    # Second PEG block: z EO units
    smi += "CCO" * z

    # Now we end with an O; to make it HO-...-OH, we just leave the final O
    # with implicit H (RDKit will treat terminal O as -OH if valence allows).
    return smi

x = 38  # EO in first PEG
y = 29  # PO in PPG
z = 38  # EO in second PEG

smiles = peg_ppg_peg_smiles(x, y, z)
mol = Chem.MolFromSmiles(smiles)

print("SMILES length:", len(smiles))
print("Mol is valid:", mol is not None)
if mol:
    print(Chem.MolToSmiles(mol))

output_file = "05a_SMILES_P188.csv"
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["name", "smiles"])
    writer.writerow(["P188", smiles])

print(f"Saved SMILES to {output_file}")