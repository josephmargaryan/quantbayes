import Bio.PDB as PDB
from Bio.PDB import Vector, calc_dihedral

"""
Extract phi psi angles using:
    Biopython
    Own implementation.
"""

###############################
# Using the Polypeptide class #
###############################


def get_phi_psi(structure):
    """
    Calculate phi,psi dihedral angles and return lists.
    Uses the polypeptide class.

    Args:
        Bio.PDB structure
    Returns:
        (list of phi angles, list of psi angles)
    """
    # Create a list of properly connected polypeptide objects
    ppb = PDB.PPBuilder()
    pp_list = ppb.build_peptides(structure)

    # Get phi and psi angles
    phi_list = []
    psi_list = []
    # Iterate over polypeptide molecules
    for pp in pp_list:
        # Calculate phi and psi angles and unpack list and tuple
        for phi, psi in pp.get_phi_psi_list():
            # put them in the lists
            phi_list.append(phi)
            psi_list.append(psi)
    return phi_list, psi_list


######################
# Own implementation #
######################


def calc_phi_psi(res1, res2, res3):
    """
    Return a tuple of phi/psi dihedral angles.

    Args:
        res1, res2, res3: Bio.PDB residue objects
    Returns:
        (phi, psi) tuple of res2
    """
    n = res2["N"].get_vector()
    ca = res2["CA"].get_vector()
    c = res2["C"].get_vector()
    # Phi
    cp = res1["C"].get_vector()
    phi = calc_dihedral(cp, n, ca, c)
    # Psi
    nn = res3["N"].get_vector()
    psi = calc_dihedral(n, ca, c, nn)
    # Return phi, psi tuple
    return (phi, psi)


if __name__ == "__main__":

    # Create parser object
    p = PDB.PDBParser(QUIET=True)
    # Get structure
    s = p.get_structure(
        "2PTC",
        "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/data/2PTC.pdb",
    )

    # Result using Biopython
    # Print (phi, psi) of residue 2
    phi_list, psi_list = get_phi_psi(s)
    print("Biopython")
    print("\tPhi angle: %.2f" % phi_list[1])
    print("\tPsi angle: %.2f" % psi_list[1])

    # Result using own implementation
    # Print (phi, psi) of residue 2
    # Get all residues
    r = list(s.get_residues())
    # Get phi psi angles of residue 2
    phi, psi = calc_phi_psi(r[0], r[1], r[2])
    print("Own implementation")
    print("\tPhi angle: %.2f" % phi)
    print("\tPsi angle: %.2f" % psi)
