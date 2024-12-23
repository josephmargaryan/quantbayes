from Bio.PDB import PDBParser

"""
Objective: Locate residue 24 in chain E of the structure 2PTC and determine its identity.
"""

parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Locate mutation (e.g., ALA to GLY at residue 24, chain E)
for chain in structure[0]:
    for residue in chain:
        if residue.id[1] == 24 and chain.id == "E":
            print(f"Residue 24: {residue.get_resname()} (Before Mutation)")

"""
Reproduce results in PyMol

PyMOL>fetch 2PTC
PyMOL>select res24, chain E and resi 24
PyMOL>iterate res24, print(f"Residue {resi}: {resn}")
"""
