"""
Script to create fixed positions dictionary for ProteinMPNN

This script generates a dictionary that specifies which residue positions should be kept fixed
(not redesigned) during protein design. It supports two modes:
1. Directly specify positions to fix
2. Specify positions to design (everything else is fixed)

Output format: {"PDB_NAME": {"chain_A": [pos1, pos2, ...], "chain_B": [...]}}
Example: {"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}}

Usage:
    # Fix specific positions (default mode):
    python make_fixed_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                        --output_path fixed_pos.jsonl \\
                                        --chain_list "A B" \\
                                        --position_list "1 2 3,4 5 6"

    # Specify positions to design (inverted mode):
    python make_fixed_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                        --output_path fixed_pos.jsonl \\
                                        --chain_list "A" \\
                                        --position_list "10 11 12" \\
                                        --specify_non_fixed
"""

import argparse


def main(args):
    """
    Main function to create fixed positions dictionary

    Args:
        args: Command line arguments containing:
            - input_path: Path to parsed PDB JSONL file
            - output_path: Path to save the fixed positions dictionary
            - chain_list: Space-separated list of chain IDs
            - position_list: Comma-separated groups of positions for each chain
            - specify_non_fixed: If True, position_list specifies positions to DESIGN
                                 (all others are fixed). If False, position_list
                                 specifies positions to FIX.
    """
    import glob
    import random
    import numpy as np
    import json
    import itertools

    # Read input JSONL file containing parsed PDB structures
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)

    # Parse position lists for each chain
    # Example: "1 2 3,4 5 6" -> [[1, 2, 3], [4, 5, 6]] for chains A and B respectively
    fixed_list = [[int(item) for item in one.split()]
                  for one in args.position_list.split(",")]

    # Parse chain list
    global_designed_chain_list = [
        str(item) for item in args.chain_list.split()
    ]

    my_dict = {}

    if not args.specify_non_fixed:
        # Default mode: position_list specifies positions to FIX
        for json_str in json_list:
            result = json.loads(json_str)

            # Extract all chain IDs from the PDB structure
            all_chain_list = [
                item[-1:] for item in list(result) if item[:9] == 'seq_chain'
            ]

            # Create dictionary for fixed positions
            fixed_position_dict = {}

            # Assign fixed positions to each chain in the design list
            for i, chain in enumerate(global_designed_chain_list):
                fixed_position_dict[chain] = fixed_list[i]

            # Chains not in the design list get empty fixed position lists
            for chain in all_chain_list:
                if chain not in global_designed_chain_list:
                    fixed_position_dict[chain] = []

            my_dict[result['name']] = fixed_position_dict

    else:
        # Inverted mode: position_list specifies positions to DESIGN
        # All other positions are fixed
        for json_str in json_list:
            result = json.loads(json_str)

            # Extract all chain IDs
            all_chain_list = [
                item[-1:] for item in list(result) if item[:9] == 'seq_chain'
            ]

            fixed_position_dict = {}

            for chain in all_chain_list:
                # Get the length of this chain's sequence
                seq_length = len(result[f'seq_chain_{chain}'])

                # Create list of all residue positions (1-indexed)
                all_residue_list = (np.arange(seq_length) + 1).tolist()

                if chain not in global_designed_chain_list:
                    # If chain is not in design list, fix all positions
                    fixed_position_dict[chain] = all_residue_list
                else:
                    # If chain is in design list, fix all positions EXCEPT those specified
                    idx = np.argwhere(
                        np.array(global_designed_chain_list) == chain)[0][0]

                    # Calculate fixed positions as: all_positions - design_positions
                    fixed_position_dict[chain] = list(
                        set(all_residue_list) - set(fixed_list[idx]))

            my_dict[result['name']] = fixed_position_dict

    # Write output dictionary to file
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')

    # Example output:
    # {"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path",
                           type=str,
                           help="Path to the parsed PDBs in JSONL format")
    argparser.add_argument("--output_path",
                           type=str,
                           help="Path to save the output fixed positions dictionary")
    argparser.add_argument("--chain_list",
                           type=str,
                           default='',
                           help="Space-separated list of chain IDs (e.g., 'A B C')")
    argparser.add_argument(
        "--position_list",
        type=str,
        default='',
        help=
        "Comma-separated position lists for each chain (e.g., '11 12 14 18,1 2 3 4' for first and second chain)")
    argparser.add_argument(
        "--specify_non_fixed",
        action="store_true",
        default=False,
        help=
        "If set, position_list specifies residues to DESIGN (all others are fixed). Default: False")

    args = argparser.parse_args()
    main(args)
