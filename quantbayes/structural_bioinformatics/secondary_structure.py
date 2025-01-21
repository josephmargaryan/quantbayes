from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

"""
To use DSSP directly in the terminal:
mkdssp 2PTC.pdb 2PTC.dssp 
"""

# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)
model = structure[0]

# Run DSSP
dssp = DSSP(
    model,
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Iterate through DSSP results
print("Residues and Secondary Structures in Chain E:")
for key in dssp.keys():
    chain_id, residue_id = key
    ss = dssp[key][2]  # Secondary structure
    if chain_id == "E":  # Filter for chain E
        print(f"Residue {residue_id} has secondary structure: {ss}")

# Print hydrogen bond information
print("\nHydrogen Bonds for Residues in Chain E:")
for key in dssp.keys():
    chain_id, residue_id = key
    if chain_id == "E":
        hbonds = dssp[key][6:8]  # Hydrogen bond information
        print(f"Residue {residue_id}: H-Bonds: {hbonds[0]}, {hbonds[1]}")


"""
Visual conformation of T:

fetch 2PTC
dss
color blue, ss h
color yellow, ss s
color green, ss t
select residue_244, chain E and resi 244
show sticks, residue_244
distance hbonds_res244, (resi 244 and chain E), (resi 240 and chain E)
show surface, chain E
get_area chain E
"""
