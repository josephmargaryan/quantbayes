from Bio.PDB import PDBParser, calc_dihedral
import matplotlib.pyplot as plt
import math

parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

phi_psi_angles = []

for chain in structure[0]:  # Iterate through all chains in the model
    residues = list(chain)  # Convert chain to a list for sequential access
    for i in range(1, len(residues) - 1):  # Skip the first and last residues
        prev_res = residues[i - 1]
        residue = residues[i]
        next_res = residues[i + 1]

        # Ensure required atoms are present in the residues
        if (
            "C" in prev_res
            and "N" in residue
            and "CA" in residue
            and "C" in residue
            and "N" in next_res
        ):
            phi = calc_dihedral(
                prev_res["C"].get_vector(),
                residue["N"].get_vector(),
                residue["CA"].get_vector(),
                residue["C"].get_vector(),
            )
            psi = calc_dihedral(
                residue["N"].get_vector(),
                residue["CA"].get_vector(),
                residue["C"].get_vector(),
                next_res["N"].get_vector(),
            )
            phi_deg = phi * 180 / math.pi
            psi_deg = psi * 180 / math.pi

            # Print phi and psi for residue ALA-24
            if (
                residue.get_resname() == "ALA"
                and residue.id[1] == 24
                and chain.id == "E"
            ):
                print(
                    f"Residue ALA-24 (Chain E): Phi: {phi_deg:.2f}, Psi: {psi_deg:.2f}"
                )

            phi_psi_angles.append((phi_deg, psi_deg))  # Convert to degrees


# Plot the Ramachandran plot
if phi_psi_angles:
    phi, psi = zip(*phi_psi_angles)
    plt.scatter(phi, psi, alpha=0.5)
    plt.xlabel("Phi (°)")
    plt.ylabel("Psi (°)")
    plt.title("Ramachandran Plot")
    plt.grid()
    plt.show()
else:
    print("No valid phi/psi angles were found.")


"""
To do the same in PyMol:
PyMOL> fetch 2PTC
PyMOL> select residue24, chain E and resi 24
PyMOL> zoom residue24
PyMOL> phi_psi residue24


Check ramanchandran plot at PROCHECK
"""
