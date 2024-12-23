from Bio.PDB import NeighborSearch, PDBParser

parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Find residues within 4Å of a given residue
residue_of_interest = None
for chain in structure[0]:
    for residue in chain:
        if residue.id[1] == 24 and chain.id == "E":  # Residue 24, Chain E
            residue_of_interest = residue

if residue_of_interest:
    atom_list = list(structure[0].get_atoms())
    ns = NeighborSearch(atom_list)
    nearby_atoms = ns.search(residue_of_interest["CA"].coord, 4.0)
    nearby_residues = {atom.get_parent() for atom in nearby_atoms}
    print(f"Residues within 4Å of residue 24 in chain E:")
    for res in nearby_residues:
        print(res)

if residue_of_interest:
    atom_list = list(structure[0].get_atoms())
    ns = NeighborSearch(atom_list)
    nearby_atoms = ns.search(residue_of_interest["CA"].coord, 4.0)
    print(f"Atoms within 4Å of residue 24 in chain E:")
    for atom in nearby_atoms:
        parent_res = atom.get_parent()
        print(
            f"Atom: {atom.get_name()}, Residue: {parent_res.get_resname()}-{parent_res.id[1]}, Chain: {parent_res.get_full_id()[2]}, Distance: {residue_of_interest['CA'] - atom:.2f}Å"
        )


"""
To do the same in PyMol

PyMOL> fetch 2PTC
PyMOL> select residue24, chain E and resi 24
PyMOL> select nearby_atoms, (resi 24 and chain E) around 4
PyMOL> select nearby_residues, byres nearby_atoms
PyMOL> show sticks, nearby_residues
PyMOL> color red, nearby_residues
PyMOL> iterate nearby_residues, print(f"{resn}-{resi} in chain {chain}")

"""
