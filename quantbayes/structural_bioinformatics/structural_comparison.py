from Bio.PDB import Superimposer, PDBParser

# Initialize PDB parser
parser = PDBParser()

# Parse the structures
structure_1 = parser.get_structure(
    "apo",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/1TPO.pdb",
)
structure_2 = parser.get_structure(
    "holo",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Select chains
chain_1 = structure_1[0]["A"]  # Chain 'A' from 1TPO
chain_2 = structure_2[0]["E"]  # Chain 'E' from 2PTC

# Extract CA atoms
atoms_1 = [res["CA"] for res in chain_1 if "CA" in res]
atoms_2 = [res["CA"] for res in chain_2 if "CA" in res]

# Ensure the same number of atoms
if len(atoms_1) != len(atoms_2):
    print("Warning: Chains have different numbers of CA atoms.")
    min_len = min(len(atoms_1), len(atoms_2))
    atoms_1 = atoms_1[:min_len]
    atoms_2 = atoms_2[:min_len]

# Superimpose
sup = Superimposer()
sup.set_atoms(atoms_1, atoms_2)
sup.apply(structure_2.get_atoms())

# Print RMSD
print(f"RMSD: {sup.rms:.2f}")


"""
Reprocude the results using PyMol:

PyMOL>fetch 2PTC
PyMOL>align 1TPO and chain A, 2PTC and chain E
 Match: read scoring matrix.
 Match: assigning 308 x 346 pairwise scores.
 MatchAlign: aligning residues (308 vs 346)...
 MatchAlign: score 1193.000
 ExecutiveAlign: 1630 atoms aligned.
 ExecutiveRMS: 78 atoms rejected during cycle 1 (RMSD=0.62).
 ExecutiveRMS: 63 atoms rejected during cycle 2 (RMSD=0.39).
 ExecutiveRMS: 36 atoms rejected during cycle 3 (RMSD=0.34).
 ExecutiveRMS: 13 atoms rejected during cycle 4 (RMSD=0.33).
 ExecutiveRMS: 13 atoms rejected during cycle 5 (RMSD=0.32).
 Executive: RMSD =    0.316 (1427 to 1427 atoms)

"""
