import Bio.PDB as PDB

"""
Simple script to save all atoms in a sphere around 
the center of a molecule to a PDB file.
"""


class CenterSelect(PDB.Select):
    def __init__(self, center):
        """
        Specify center.

        Args:
            center: numpy array, shape (3,)
        """
        self.center = center

    def accept_atom(self, atom):
        """Accept atoms close to center"""
        # Distance = length (ie. norm) of difference vector
        diff = self.center - atom.get_vector()
        dist = diff.norm()
        if dist < 10:
            # Close
            return 1
        else:
            # Not close
            return 0


if __name__ == "__main__":

    # Create parser
    p = PDB.PDBParser(QUIET=True)
    # Get structure
    s = p.get_structure(
        "2PTC",
        "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/data/2PTC.pdb",
    )
    # Get the enzyme chain (E)
    enzyme = s[0]["E"]

    # Find CA center-of-mass
    # Atom counter
    n = 0
    # Use Bio.PDB.Vector to calculate center-of-mass
    atom_sum = PDB.Vector(0.0, 0.0, 0.0)
    for res in enzyme:
        # Check if res is sane (not water or missing CA)
        if res.has_id("CA"):
            # In-place elementwise addition
            atom_sum += res["CA"].get_vector()
            n += 1
    # Divide by the number of atoms to get center-of-mass
    # Note: ** multiplies Vector with Vector or scalar
    com = atom_sum ** (1 / n)

    print("Center of mass is: ", com)

    # Create PDBIO object
    io = PDB.PDBIO()
    # Set the structure
    io.set_structure(s)
    # Filename for saving
    outfile = "2ptc-center.pdb"

    # Create CenterSelect object and pass the calculated center-of-mass
    select = CenterSelect(com)

    # Save structure using CenterSelect filter
    io.save(outfile, select)

    print("Saved to %s" % outfile)
