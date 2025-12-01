"""
Script to create per-residue amino acid bias dictionary for ProteinMPNN

This script generates a dictionary that specifies amino acid biases at each residue position
in each chain. This allows fine-grained control over amino acid preferences at specific
positions during sequence design.

Key features:
- Position-specific amino acid biases
- Different biases for different chains
- Positive bias values make specific amino acids more likely
- Large positive values (e.g., 100.5) effectively force specific amino acids
- Large negative values (e.g., -100.5) effectively prohibit specific amino acids

The bias matrix has shape [chain_length, 21] where 21 corresponds to the 20 standard
amino acids plus X (any amino acid). The order is: ACDEFGHIKLMNPQRSTVWYX

Output format: {"PDB_NAME": {"chain_A": [[bias_matrix]], "chain_B": [[bias_matrix]]}}
Example: {"5TTA": {"A": [[0.0, 0.0, ...], [0.0, 0.0, ...]], "B": [[...]]}}

Usage:
    python make_bias_per_res_dict.py --input_path parsed_pdbs.jsonl \\
                                      --output_path bias_per_res.jsonl

Note: This is a template script. You need to modify the hardcoded residue and
amino acid specifications within the script to match your design needs.
"""

import argparse


def main(args):
    """
    Main function to create per-residue amino acid bias dictionary

    Args:
        args: Command line arguments containing:
            - input_path: Path to the parsed PDB JSONL file
            - output_path: Path to save the per-residue bias dictionary

    Note: The actual bias specifications are hardcoded in the script and need to be
    modified based on your specific design requirements.
    """
    import glob
    import random
    import numpy as np
    import json

    # ProteinMPNN amino acid alphabet (21 characters: 20 standard AAs + X)
    mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    # Mapping from single-letter amino acid codes to indices
    # This defines the order used in the bias matrix
    mpnn_alphabet_dict = {
        'A': 0,   # Alanine
        'C': 1,   # Cysteine
        'D': 2,   # Aspartic acid
        'E': 3,   # Glutamic acid
        'F': 4,   # Phenylalanine
        'G': 5,   # Glycine
        'H': 6,   # Histidine
        'I': 7,   # Isoleucine
        'K': 8,   # Lysine
        'L': 9,   # Leucine
        'M': 10,  # Methionine
        'N': 11,  # Asparagine
        'P': 12,  # Proline
        'Q': 13,  # Glutamine
        'R': 14,  # Arginine
        'S': 15,  # Serine
        'T': 16,  # Threonine
        'V': 17,  # Valine
        'W': 18,  # Tryptophan
        'Y': 19,  # Tyrosine
        'X': 20   # Any amino acid
    }

    # Read input JSONL file containing parsed PDB structures
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)

    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)

        # Extract all chain IDs from the PDB structure
        all_chain_list = [
            item[-1:] for item in list(result) if item[:10] == 'seq_chain_'
        ]

        # Dictionary to store bias matrices for each chain
        bias_by_res_dict = {}

        for chain in all_chain_list:
            # Get the length of the current chain
            chain_length = len(result[f'seq_chain_{chain}'])

            # Initialize bias matrix with zeros (no bias)
            # Shape: [chain_length, 21] where 21 = number of amino acids
            bias_per_residue = np.zeros([chain_length, 21])

            # Example bias specification for chain A
            # MODIFY THIS SECTION based on your specific design requirements
            if chain == 'A':
                # Specify which residue positions to bias (0-indexed)
                residues = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]

                # Specify which amino acids to favor at these positions
                amino_acids = [5, 9]  # [G, L] - Glycine and Leucine

                # Apply positive bias to favor these amino acids at these positions
                for res in residues:
                    for aa in amino_acids:
                        # Large positive bias (100.5) effectively forces these amino acids
                        bias_per_residue[res, aa] = 100.5

            # Example bias specification for chain C
            # MODIFY THIS SECTION based on your specific design requirements
            if chain == 'C':
                # Specify which residue positions to bias (0-indexed)
                residues = [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15]

                # Specify which amino acids to disfavor at these positions
                # range(21)[1:] gives indices 1-20, excluding index 0 (Alanine)
                amino_acids = range(21)[1:]

                # Apply negative bias to disfavor these amino acids at these positions
                for res in residues:
                    for aa in amino_acids:
                        # Large negative bias (-100.5) effectively prohibits these amino acids
                        bias_per_residue[res, aa] = -100.5

            # Convert numpy array to list for JSON serialization
            bias_by_res_dict[chain] = bias_per_residue.tolist()

        # Store bias dictionary for this PDB structure
        my_dict[result['name']] = bias_by_res_dict

    # Write the output dictionary to a JSONL file
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path",
                           type=str,
                           help="Path to the parsed PDBs in JSONL format")
    argparser.add_argument("--output_path",
                           type=str,
                           help="Path to save the output per-residue bias dictionary")

    args = argparser.parse_args()
    main(args)

# Example output:
# {"PDB_NAME": {"A": [[0.0, 0.0, ..., 0.0], ...], "B": [[...]]}}
# Each inner list represents the bias values for the 21 amino acids at a specific position
