"""
Script to create tied positions dictionary with positive/negative design weights for ProteinMPNN

This script generates a dictionary that specifies which residue positions should have
their amino acid identities tied together during protein design, with additional control
over positive and negative design weights. This is useful for:
1. Designing homooligomers with weighted chain contributions
2. Positive design: encouraging specific sequences (weight = 1.0)
3. Negative design: discouraging specific sequences (weight = -0.1 to -0.5)
4. Neutral: ignoring chain information (weight = 0.0)

Output format: {"PDB_NAME": [{"chain_A": [[pos], [weight]], "chain_B": [[pos], [weight]]}, ...]}
Example: {"5TTA": [], "3LIS": [{"A": [[1], [1.0]], "B": [[1], [-0.1]]}, ...]}

Usage:
    # For non-homooligomeric design with specific tied positions:
    python make_pos_neg_tied_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                                 --output_path tied_pos.jsonl \\
                                                 --chain_list "A B" \\
                                                 --position_list "1 2 3,1 2 3" \\
                                                 --homooligomer 0

    # For homooligomeric design with positive/negative weights:
    python make_pos_neg_tied_positions_dict.py --input_path parsed_pdbs.jsonl \\
                                                 --output_path tied_pos.jsonl \\
                                                 --homooligomer 1 \\
                                                 --pos_neg_chain_list "A B,C D" \\
                                                 --pos_neg_chain_betas "1.0 1.0,-0.5 -0.5"
"""

import argparse


def main(args):
    """
    Main function to create tied positions dictionary with positive/negative design weights

    Args:
        args: Command line arguments containing:
            - input_path: Path to the parsed PDB JSONL file
            - output_path: Path to save the tied positions dictionary
            - chain_list: Space-separated list of chains to tie together
            - position_list: Comma-separated position lists for each chain
            - homooligomer: If 0, use specified positions; if 1, tie all positions
                           across all chains for homooligomer design
            - pos_neg_chain_list: Comma-separated groups of chains to tie together
            - pos_neg_chain_betas: Weight values for each chain group (1.0 for positive design,
                                   -0.1 to -0.5 for negative design, 0.0 to ignore)
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
        # Mode 1: Homooligomer design with positive/negative design weights
        # Parse chain groups and their corresponding beta weights if provided
        if args.pos_neg_chain_list:
            # Parse chain list groups
            # Example: "A B,C D" -> [['A', 'B'], ['C', 'D']]
            chain_list_input = [[str(item) for item in one.split()]
                                for one in args.pos_neg_chain_list.split(",")]

            # Parse beta weight groups
            # Example: "1.0 1.0,-0.5 -0.5" -> [[1.0, 1.0], [-0.5, -0.5]]
            chain_betas_input = [[float(item) for item in one.split()]
                                 for one in args.pos_neg_chain_betas.split(",")
                                 ]

            # Flatten the nested lists for easier lookup
            chain_list_flat = [
                item for sublist in chain_list_input for item in sublist
            ]
            chain_betas_flat = [
                item for sublist in chain_betas_input for item in sublist
            ]

            # Create dictionary mapping chains to their beta weights
            # Beta values: 1.0 (positive design), -0.1 to -0.5 (negative design), 0.0 (ignore)
            chain_betas_dict = dict(zip(chain_list_flat, chain_betas_flat))

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

            # Process each group of chains separately
            for chains in chain_list_input:
                # Tie each position across chains in this group
                for i in range(1, chain_length + 1):
                    temp_dict = {}
                    # For each chain in the current group
                    for j, chain in enumerate(chains):
                        if args.pos_neg_chain_list and chain in chain_list_flat:
                            # Use specified beta weight for this chain
                            # Format: [[residue_number], [beta_weight]]
                            # First list contains residue numbers to tie
                            # Second list contains weights for positive/negative design
                            temp_dict[chain] = [[i], [chain_betas_dict[chain]]]
                        else:
                            # Default to positive design with weight 1.0
                            temp_dict[chain] = [
                                [i], [1.0]
                            ]  # First list: residue numbers, second list: energy weights
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
    argparser.add_argument("--pos_neg_chain_list",
                           type=str,
                           default='',
                           help="Comma-separated chain groups to be tied together (e.g., 'A B,C D')")
    argparser.add_argument(
        "--pos_neg_chain_betas",
        type=str,
        default='',
        help=
        "Beta weight values for each chain group (e.g., '1.0 1.0,-0.5 -0.5'). Use 1.0 for positive design, -0.1 to -0.5 for negative design, 0.0 to ignore chain")

    args = argparser.parse_args()
    main(args)

# Example output for homooligomer mode with positive/negative design:
# {"5TTA": [], "3LIS": [{"A": [[1], [1.0]], "B": [[1], [-0.1]]}, {"A": [[2], [1.0]], "B": [[2], [-0.1]]}, ...]}
# Position 1 in chain A is positively designed (weight 1.0), chain B is negatively designed (weight -0.1)
