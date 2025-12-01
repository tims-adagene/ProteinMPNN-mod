"""
Script to parse PDB files with multiple chains for ProteinMPNN

This script parses PDB files and extracts structural information (coordinates and sequences)
for all chains. It handles:
- Multiple chains per structure
- Both full-atom and CA-only (C-alpha backbone) representations
- MSE (selenomethionine) residues (converted to MET)
- Missing residues and atoms
- Insertion codes in residue numbering

The output is a JSONL file where each line is a JSON object containing all structural
information for one PDB file.

Output format per PDB:
{
    "name": "PDB_NAME",
    "num_of_chains": N,
    "seq": "concatenated_sequence",
    "seq_chain_A": "sequence_A",
    "seq_chain_B": "sequence_B",
    "coords_chain_A": {"N_chain_A": [...], "CA_chain_A": [...], "C_chain_A": [...], "O_chain_A": [...]},
    "coords_chain_B": {...},
    ...
}

Usage:
    # For full-atom parsing (N, CA, C, O):
    python parse_multiple_chains.py --input_path /path/to/pdbs/ \\
                                     --output_path parsed_pdbs.jsonl

    # For CA-only parsing:
    python parse_multiple_chains.py --input_path /path/to/pdbs/ \\
                                     --output_path parsed_pdbs.jsonl \\
                                     --ca_only
"""

import argparse


def main(args):
    """
    Main function to parse PDB files and extract structural information

    Args:
        args: Command line arguments containing:
            - input_path: Path to folder containing PDB files
            - output_path: Path to save the parsed structures in JSONL format
            - ca_only: If True, only parse CA atoms; if False, parse N, CA, C, O
    """

    import numpy as np
    import os, time, gzip, json
    import glob

    folder_with_pdbs_path = args.input_path
    save_path = args.output_path
    ca_only = args.ca_only

    # Define amino acid code mappings
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")  # Single-letter codes + gap
    states = len(alpha_1)

    # Three-letter amino acid codes
    alpha_3 = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'GAP'
    ]

    # Create conversion dictionaries between different amino acid representations
    aa_1_N = {a: n for n, a in enumerate(alpha_1)}  # Single letter -> index
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}  # Three letter -> index
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}  # Index -> single letter
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}  # Single -> three letter
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}  # Three -> single letter

    def AA_to_N(x):
        """
        Convert amino acid sequences to numeric indices

        Args:
            x: List of amino acid sequences (e.g., ["ARND", "ACDE"])

        Returns:
            List of lists of numeric indices (e.g., [[0,1,2,3], [0,1,2,3]])
        """
        x = np.array(x)
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        """
        Convert numeric indices back to amino acid sequences

        Args:
            x: Array of numeric indices (e.g., [[0,1,2,3]])

        Returns:
            List of amino acid sequences (e.g., ["ARND"])
        """
        x = np.array(x)
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
        """
        Parse a PDB file and extract coordinates and sequence for specified chain

        Args:
            x: Path to PDB file
            atoms: List of atom names to extract (default: ['N', 'CA', 'C'])
            chain: Chain ID to parse (if None, parse all chains)

        Returns:
            Tuple of (coordinates, sequence):
            - coordinates: numpy array of shape [L, num_atoms, 3] containing (x,y,z) coords
            - sequence: list with one string element containing the amino acid sequence

        Special handling:
            - MSE (selenomethionine) is converted to MET (methionine)
            - Missing residues are filled with gaps
            - Missing atoms are filled with NaN values
            - Insertion codes (e.g., 100A) are handled
        """
        xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6

        # Read and parse PDB file line by line
        for line in open(x, "rb"):
            line = line.decode("utf-8", "ignore").rstrip()

            # Convert MSE (selenomethionine) to MET (methionine)
            if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            # Process ATOM records
            if line[:4] == "ATOM":
                ch = line[21:22]  # Chain ID

                # Only process if this is the target chain or parsing all chains
                if ch == chain or chain is None:
                    atom = line[12:12 + 4].strip()  # Atom name
                    resi = line[17:17 + 3]  # Residue name (3-letter code)
                    resn = line[22:22 + 5].strip()  # Residue number + insertion code

                    # Parse coordinates
                    x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                    # Handle insertion codes (e.g., "100A" -> resa='A', resn=99)
                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1

                    # Track min and max residue numbers for gap filling
                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn

                    # Initialize nested dictionaries if needed
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    # Store atom coordinates
                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

        # Convert parsed data to numpy arrays and fill in missing values
        seq_, xyz_ = [], []
        try:
            # Iterate through all residue positions from min to max
            for resn in range(min_resn, max_resn + 1):
                if resn in seq:
                    # Residue exists - add its sequence
                    for k in sorted(seq[resn]):
                        seq_.append(aa_3_N.get(seq[resn][k], 20))
                else:
                    # Missing residue - add gap
                    seq_.append(20)

                if resn in xyz:
                    # Residue exists - extract atom coordinates
                    for k in sorted(xyz[resn]):
                        for atom in atoms:
                            if atom in xyz[resn][k]:
                                xyz_.append(xyz[resn][k][atom])
                            else:
                                # Missing atom - add NaN coordinates
                                xyz_.append(np.full(3, np.nan))
                else:
                    # Missing residue - add NaN for all atoms
                    for atom in atoms:
                        xyz_.append(np.full(3, np.nan))

            # Return coordinates reshaped to [L, num_atoms, 3] and sequence
            return np.array(xyz_).reshape(-1, len(atoms),
                                          3), N_to_AA(np.array(seq_))
        except TypeError:
            # Return error strings if parsing fails
            return 'no_chain', 'no_chain'

    pdb_dict_list = []
    c = 0

    # Ensure folder path ends with '/'
    if folder_with_pdbs_path[-1] != '/':
        folder_with_pdbs_path = folder_with_pdbs_path + '/'

    # Define chain alphabet (supports up to 300+ chains)
    # Standard chains: A-Z, a-z
    # Extended chains: 0-299
    init_alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    # Get list of all PDB files in the folder
    biounit_names = glob.glob(folder_with_pdbs_path + '*.pdb')

    # Process each PDB file
    for biounit in biounit_names:
        my_dict = {}
        s = 0  # Chain counter
        concat_seq = ''  # Concatenated sequence of all chains
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}

        # Try to parse each possible chain ID
        for letter in chain_alphabet:
            # Determine which atoms to parse based on ca_only flag
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']

            # Parse the current chain
            xyz, seq = parse_PDB_biounits(biounit,
                                          atoms=sidechain_atoms,
                                          chain=letter)

            # Check if chain exists (not an error string)
            if type(xyz) != str:
                # Chain exists - store its data
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]

                coords_dict_chain = {}

                if ca_only:
                    # Store only CA coordinates
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    # Store N, CA, C, O coordinates separately
                    coords_dict_chain['N_chain_' +
                                      letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' +
                                      letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' +
                                      letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' +
                                      letter] = xyz[:, 3, :].tolist()

                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1  # Increment chain counter

        # Extract PDB name from file path
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq

        # Only add to output if we found at least one chain
        # and didn't exhaust the chain alphabet
        if s < len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1

    # Write all parsed PDB structures to JSONL file
    # Each line is a separate JSON object
    with open(save_path, 'w') as f:
        for entry in pdb_dict_list:
            f.write(json.dumps(entry) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        "--input_path",
        type=str,
        help="Path to folder containing PDB files (e.g., /home/my_pdbs/)")
    argparser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the parsed structures in JSONL format")
    argparser.add_argument(
        "--ca_only",
        action="store_true",
        default=False,
        help="Parse only C-alpha backbone atoms (default: False, parse N/CA/C/O)")

    args = argparser.parse_args()
    main(args)

# Example output (one line per PDB, shown formatted for readability):
# {
#   "name": "1ABC",
#   "num_of_chains": 2,
#   "seq": "ARNDCQEGHILKMFPSTWYVARNDCQEGHILKMFPSTWYV",
#   "seq_chain_A": "ARNDCQEGHILKMFPSTWYV",
#   "seq_chain_B": "ARNDCQEGHILKMFPSTWYV",
#   "coords_chain_A": {
#     "N_chain_A": [[x, y, z], [x, y, z], ...],
#     "CA_chain_A": [[x, y, z], [x, y, z], ...],
#     "C_chain_A": [[x, y, z], [x, y, z], ...],
#     "O_chain_A": [[x, y, z], [x, y, z], ...]
#   },
#   "coords_chain_B": {...}
# }
