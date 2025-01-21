from Bio.PDB import PDBParser, NeighborSearch

"""
Exercise 1:
Find all trypsin (chain E) /trypsin inhibitor (chain I) residue pairs that have more
than two close contacts.
    - A “close contact” is any atom pair with distance lower than 3.5 Å.


"""
# Parse the PDB file
parser = PDBParser()
structure = parser.get_structure(
    "2ptc",
    "/Users/josephmargaryan/Desktop/probabilistic_ML/structural_bioinformatics/2PTC.pdb",
)

# Extract chains
chain_E = structure[0]["E"]  # Trypsin
chain_I = structure[0]["I"]  # Trypsin Inhibitor

# Collect all atoms for neighbor search
atoms_E = list(chain_E.get_atoms())
atoms_I = list(chain_I.get_atoms())

# Create a neighbor search for chain E atoms
ns_E = NeighborSearch(atoms_E)

# Find residue pairs with more than 2 close contacts
close_contacts = []
for residue_I in chain_I:
    contact_count = {}
    for atom_I in residue_I:
        nearby_atoms = ns_E.search(atom_I.coord, 3.5)  # 3.5 Å threshold
        for atom in nearby_atoms:
            residue_E = atom.get_parent()
            contact_count[residue_E] = contact_count.get(residue_E, 0) + 1

    for residue_E, count in contact_count.items():
        if count > 2:
            close_contacts.append((residue_E, residue_I, count))

# Print results
for residue_E, residue_I, count in close_contacts:
    print(f"{residue_E} - {residue_I}: {count} close contacts")

for residue_E, residue_I, count in close_contacts:
    print(
        f"{residue_E.get_resname()}{residue_E.get_id()[1]} - {residue_I.get_resname()}{residue_I.get_id()[1]}: {count} close contacts"
    )


"""
Reproduce results in PyMol:

PyMOL>fetch 2PTC
TITLE     THE GEOMETRY OF THE REACTIVE SITE AND OF THE PEPTIDE GROUPS IN TRYPSIN, TRYPSINOGEN AND ITS COMPLEXES WITH INHIBITORS
 ExecutiveLoad-Detail: Detected mmCIF
 CmdLoad: "./2ptc.cif" loaded as "2PTC".
PyMOL>select trypsin_chain_E, chain E
 Selector: selection "trypsin_chain_E" defined with 1752 atoms.
PyMOL>select inhibitor_chain_I, chain I
 Selector: selection "inhibitor_chain_I" defined with 489 atoms.
PyMOL>color red, trypsin_chain_E
 Executive: Colored 1752 atoms.
PyMOL>color blue, inhibitor_chain_I
 Executive: Colored 489 atoms.
PyMOL>show sticks, trypsin_chain_E or inhibitor_chain_I
PyMOL>select ser195, chain E and resi 195
 Selector: selection "ser195" defined with 6 atoms.
PyMOL>select lys15, chain I and resi 15
 Selector: selection "lys15" defined with 9 atoms.
PyMOL>show sticks, ser195 or lys15
PyMOL>zoom ser195 or lys15
PyMOL>select close_contacts, (ser195 around 3.5) and lys15
 Selector: selection "close_contacts" defined with 5 atoms.
PyMOL>show spheres, close_contacts
PyMOL>distance close_contacts_dist, ser195, lys15
 Executive: object "close_contacts_dist" created.
select lys15_close_contacts, (lys15 within 3.5 of ser195)
iterate lys15_close_contacts, print(f"Lys15 - Chain: {chain}, Residue: {resi}, Atom: {name}")


"""
