from Bio.PDB import PDBParser, calc_angle, calc_dihedral, PPBuilder
from Bio.PDB.DSSP import DSSP
import math

# Download files
# !curl -O https://files.rcsb.org/download/1MBN.pdb

# Parse the structure
pdb_file = (
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb"
)
parser = PDBParser()
structure = parser.get_structure("2PTC", pdb_file)
model = structure[0]

# Finding polypeptides
ppb = PPBuilder()
pp_list = ppb.build_peptides(model)
for pp in pp_list:
    print("finding polypeptides")
    print(pp)

# Print sequence of first polypeptide
pp1 = pp_list[0]
print(pp1.get_sequence())
# Get phi, psi list
pp_list = pp1.get_phi_psi_list()


"""
for atom in model.get_atoms():
    print(atom)

for res in model.get_residues():
    print(res)  

atom_list=list(model.get_atoms())
"""

# Iterate through structure
for model in structure:
    # print(model.get_id())
    for chain in model:
        # print(chain.get_id() )
        for residue in chain:
            # print(residue.is_disordered())
            # hf, si, ic=residue.get_id()
            # print(residue.get_id())
            # print(residue.get_resname())
            for atom in residue:
                # print(atom)
                # print(atom.get_id())
                # Atom name
                # print(atom.get_name())
                # Temperature factor
                # print(atom.get_bfactor())
                # Coordinates as numpy array
                # print(atom.get_coord())
                # Coordinates as Vector object
                # print(atom.get_vector())
                # Alternative location specifier
                # print(atom.get_altloc())
                break


# Filtering for specific atoms

atom1 = None
atom2 = None
atom3 = None

for model in structure:
    for chain in model:
        for residue in chain:
            if residue.get_resname() == "ALA":
                for atom in residue:
                    if atom.get_name() == "CA":
                        atom1 = atom
                    elif atom.get_name() == "N":
                        atom2 = atom
                    elif atom.get_name() == "C":
                        atom3 = atom

                    # Break out of the loop
                    if atom1 and atom2 and atom3:
                        break
                if atom1 and atom2 and atom3:
                    break
            if atom1 and atom2 and atom3:
                break
        if atom1 and atom2 and atom3:
            break

if atom1 and atom2 and atom3:
    atom1_vector = atom1.get_vector()
    atom2_vector = atom2.get_vector()
    atom3_vector = atom3.get_vector()

    # Calculate angles
    angle = calc_angle(atom1_vector, atom2_vector, atom3_vector)
    print(
        f"Atom 1: {atom1.get_name()}, Residue: {atom1.get_parent().get_resname()}, Chain: {atom1.get_parent().get_full_id()[2]}, Residue ID: {atom1.get_parent().get_id()}"
    )
    print(
        f"Atom 2: {atom2.get_name()}, Residue: {atom2.get_parent().get_resname()}, Chain: {atom2.get_parent().get_full_id()[2]}, Residue ID: {atom2.get_parent().get_id()}"
    )
    print(
        f"Atom 3: {atom3.get_name()}, Residue: {atom3.get_parent().get_resname()}, Chain: {atom3.get_parent().get_full_id()[2]}, Residue ID: {atom3.get_parent().get_id()}"
    )
    print("----" * 20, "\n")
    print(f"Angle: {angle}")
    angle_in_degrees = math.degrees(0.5772569260539201)
    print(f"Angle in degrees: {angle_in_degrees:.2f}")

    # We need four atoms to calculate dihedral angles
    # dihedral_angle = calc_dihedral(atom1_vector, atom2_vector, atom3_vector)
    # print(dihedral_angle)

"""
To do the same in PyMol:
>>> fetch 2ptc
>>> select atom1, chain E and resi 24 and name CA
>>> select atom2, chain E and resi 24 and name N
>>> select atom3, chain E and resi 24 and name C
>>> angle result_name, atom1, atom2, atom3

"""
"""
# Run DSSP
dssp = DSSP(model, pdb_file, dssp='mkdssp')  # Adjust `dssp` executable if necessary

# Access DSSP data
for key, values in dssp.property_dict.items():
    chain_id, residue_id = key
    dssp_index, aa, ss, rel_asa, phi, psi, *_ = values
    print(f"Chain {chain_id}, Residue {residue_id}, SS: {ss}, ASA: {rel_asa:.2f}, Phi: {phi:.2f}, Psi: {psi:.2f}")

"""
