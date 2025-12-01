"""
Script to create tied positions dictionary for ProteinMPNN

This script generates a dictionary that specifies which residue positions should have
their amino acid identities tied together during protein design. This is useful for:
1. Designing homooligomers (where all chains should have identical sequences)
2. Enforcing symmetry constraints across different chains
3. Maintaining specific residue correspondences between chains

Output format: {"PDB_NAME": [{"chain_A": [pos], "chain_B": [pos]}, ...]}
Example: {"5TTA": [], "3LIS": [{"A": [1], "B": [1]}, {"A": [2], "B": [2]}, ...]}

Usage:
    # For non-homooligomeric design with specific tied positions:
    python make_tied_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                        --output_path tied_pos.jsonl \\
                                        --chain_list "A B" \\
                                        --position_list "1 2 3,1 2 3" \\
                                        --homooligomer 0

    # For homooligomeric design (all positions tied across all chains):
    python make_tied_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                        --output_path tied_pos.jsonl \\
                                        --homooligomer 1
"""

import argparse


def main(args):
    """
    Main function to create tied positions dictionary

    Args:
        args: Command line arguments containing:
            - input_path: Path to the parsed PDB JSONL file
            - output_path: Path to save the tied positions dictionary
            - chain_list: Space-separated list of chains to tie together
            - position_list: Comma-separated position lists for each chain
            - homooligomer: If 0, use specified positions; if 1, tie all positions
                           across all chains for homooligomer design
    """

    import glob
    import random
    import numpy as np
    import json
    import itertools

    # Read the input JSONL file containing parsed PDB structures
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)

    # Check if homooligomeric design mode is enabled
    homooligomeric_state = args.homooligomer

    if homooligomeric_state == 0:
        # Mode 0: Manually specify which positions to tie across chains
        # Parse position lists for each chain
        # Example: "1 2 3,1 2 3" -> [[1, 2, 3], [1, 2, 3]] for chains A and B
        tied_list = [[int(item) for item in one.split()]
                     for one in args.position_list.split(",")]

        # Parse the list of chains to be tied together
        global_designed_chain_list = [
            str(item) for item in args.chain_list.split()
        ]

        my_dict = {}
        for json_str in json_list:
            result = json.loads(json_str)

            # Extract all chain IDs from the PDB structure
            all_chain_list = sorted([
                item[-1:] for item in list(result) if item[:9] == 'seq_chain'
            ])  # Returns sorted list like ['A', 'B', 'C', ...]

            # Create list of tied position dictionaries
            # Each entry specifies which positions are tied across chains
            tied_positions_list = []

            # Iterate through positions in the first chain's tied list
            for i, pos in enumerate(tied_list[0]):
                temp_dict = {}
                # For each chain, specify the corresponding position to tie
                for j, chain in enumerate(global_designed_chain_list):
                    # Store position as a list (required format for ProteinMPNN)
                    temp_dict[chain] = [tied_list[j][i]]
                tied_positions_list.append(temp_dict)

            # Store tied positions for this PDB structure
            my_dict[result['name']] = tied_positions_list
    else:
        # Mode 1: Homooligomer design - tie ALL positions across ALL chains
        # This ensures all chains will have identical sequences
        my_dict = {}
        for json_str in json_list:
            result = json.loads(json_str)

            # Extract all chain IDs from the PDB structure
            all_chain_list = sorted([
                item[-1:] for item in list(result) if item[:9] == 'seq_chain'
            ])  # Returns sorted list like ['A', 'B', 'C', ...]

            tied_positions_list = []

            # Get the length of the first chain (all chains should be same length for homooligomer)
            chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])

            # Tie each position across all chains
            for i in range(1, chain_length + 1):
                temp_dict = {}
                # For each chain, tie the current position i
                for j, chain in enumerate(all_chain_list):
                    # Store position as a list (required format for ProteinMPNN)
                    temp_dict[chain] = [i]
                tied_positions_list.append(temp_dict)

            # Store tied positions for this PDB structure
            my_dict[result['name']] = tied_positions_list

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
                           help="Path to save the output tied positions dictionary")
    argparser.add_argument("--chain_list",
                           type=str,
                           default='',
                           help="Space-separated list of chains to tie together (e.g., 'A B C')")
    argparser.add_argument(
        "--position_list",
        type=str,
        default='',
        help=
        "Comma-separated position lists for each chain (e.g., '11 12 14 18,1 2 3 4' for first and second chain)")
    argparser.add_argument(
        "--homooligomer",
        type=int,
        default=0,
        help="If 0: use specified positions; if 1: design homooligomer (tie all positions across all chains)")

    args = argparser.parse_args()
    main(args)

# Example output for homooligomer mode (homooligomer=1):
# {"5TTA": [], "3LIS": [{"A": [1], "B": [1]}, {"A": [2], "B": [2]}, {"A": [3], "B": [3]}, ...]}
# This ensures position 1 in chain A has the same amino acid as position 1 in chain B, etc.
