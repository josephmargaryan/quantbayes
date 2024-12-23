from Bio.PDB import PDBList, calc_angle, PDBParser, calc_dihedral
import math

parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    file="/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

atom1, atom2, atom3 = None, None, None
for chain in structure[0]:
    for residue in chain:
        if residue.get_resname() == "ALA" and residue.id[1] == 24:
            atom1 = residue["CA"]
            atom2 = residue["N"]
            atom3 = residue["C"]

        if atom1 and atom2 and atom3:
            break
    if atom1 and atom2 and atom3:
        break
if atom1 and atom2 and atom3:
    angle = calc_angle(atom1.get_vector(), atom2.get_vector(), atom3.get_vector())
    print(f"The bond angle between the residues\n{angle} radians")
    print(f"The bond angle between the residues\n{math.degrees(angle)} degrees")


phi_atoms = [None] * 4
for chain in structure[0]:
    for residue in chain:
        if residue.id[1] == 23:  # Previous residue
            phi_atoms[0] = residue["C"]
        elif residue.id[1] == 24:  # Current residue
            phi_atoms[1] = residue["N"]
            phi_atoms[2] = residue["CA"]
            phi_atoms[3] = residue["C"]

# Calculate dihedral
if all(phi_atoms):
    dihedral = calc_dihedral(*[atom.get_vector() for atom in phi_atoms])
    print(f"Phi angle (degrees): {dihedral * 180 / 3.14159:.2f}")


"""
Reproduce results using PyMol:

PyMOL>fetch 2PTC
PyMOL>select ala24_ca, chain E and resi 24 and name CA
PyMOL>select ala24_n, chain E and resi 24 and name N
PyMOL>select ala24_c, chain E and resi 24 and name C
PyMOL>select gly23_c, chain E and resi 23 and name C
PyMOL>angle bond_angle, (chain E and resi 24 and name CA), (chain E and resi 24 and name N), (chain E and resi 24 and name C)

"""
