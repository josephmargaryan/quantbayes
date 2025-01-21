from Bio.PDB import Selection, PDBParser, NeighborSearch

# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Automatically detect potential ligands
ligand = None
for chain in structure[0]:
    for residue in chain:
        if residue.id[0] != " ":  # Heteroatoms (ligands, ions, water)
            print(f"Potential ligand: {residue}")
            if residue.resname == "CA":  # Update with the correct ligand name
                ligand = residue

if ligand:
    print(f"Ligand found: {ligand}")
    # Neighbor search
    atom_list = list(structure[0].get_atoms())
    ns = NeighborSearch(atom_list)

    # Choose an atom from the ligand to define the search center
    for atom in ligand:
        print(f"Ligand atom: {atom.get_name()}")

    # Perform the search
    try:
        center_atom = ligand["CA"]  # Example using CA atom
        nearby_atoms = ns.search(center_atom.coord, 5.0)  # 5Å distance
        interacting_residues = {atom.get_parent() for atom in nearby_atoms}

        print(f"Residues interacting with the ligand:")
        for res in interacting_residues:
            print(res)
    except KeyError:
        print("The specified atom is not in the ligand.")
else:
    print("Ligand not found.")

"""
Reproduce the results in PyMol:

PyMOL>fetch 2PTC
PyMOL>show spheres, ligand
PyMOL>select interacting_residues, byres (ligand around 5)
PyMOL>show sticks, interacting_residues
PyMOL>iterate interacting_residues, print(chain, resi, resn)

"""
