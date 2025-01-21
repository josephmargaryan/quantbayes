"""
First rn:
mkdssp 2PTC.pdb 2PTC.dssp   
in the terminal to create the file:
"""

with open("2PTC.dssp") as dssp_file:
    for line in dssp_file:
        if line.startswith("  #"):  # Skip header
            break
    for line in dssp_file:
        columns = line.split()
        residue_num = columns[0]
        hydrogen_bond_1 = columns[6:8]  # NH->O bond
        hydrogen_bond_2 = columns[8:10]  # O->NH bond
        print(
            f"Residue {residue_num}: NH->O: {hydrogen_bond_1}, O->NH: {hydrogen_bond_2}"
        )
