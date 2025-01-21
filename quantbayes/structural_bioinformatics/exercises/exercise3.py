import Bio.PDB as PDB

"""
Simple script to save a specific chain to a seperate PDB file.
"""


# define a class based on Bio.PDB.Select to output chain E only
class ChainSelect(PDB.Select):
    """
    Helper class for writing out a PDB file for a specific chain.
    For use with Bio.PDB's PDBIO class.
    """

    def __init__(self, chain_name):
        """
        Args:
            chain_name: string, name of chain to be written to PDB file.
        """
        self.chain_name = chain_name

    def accept_chain(self, chain):
        """
        Overload accept_chain method.
        """
        # Does the chain name fit?
        if chain.get_id() == self.chain_name:
            return True
        else:
            return False


if __name__ == "__main__":

    # Load structure
    p = PDB.PDBParser(QUIET=True)
    s = p.get_structure(
        "2PTC",
        "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/data/2PTC.pdb",
    )

    # Create PDBIO object
    io = PDB.PDBIO()

    # Set the structure
    io.set_structure(s)

    # Filename of output PDB file
    outfile = "out.pdb"

    # Create an object of class ChainSelect
    # Specify chain E
    select = ChainSelect("E")

    # Save the structure using the ChainSelect object as filter
    io.save(outfile, select)

    print("Saved to %s" % outfile)
