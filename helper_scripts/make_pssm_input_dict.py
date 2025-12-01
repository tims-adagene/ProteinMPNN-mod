"""
Script to create PSSM (Position-Specific Scoring Matrix) input dictionary for ProteinMPNN

This script reads pre-computed PSSM data from NPZ files and converts it into a JSONL
dictionary format that can be used by ProteinMPNN. PSSMs provide position-specific amino
acid preferences based on evolutionary information or other sources.

Key features:
- Loads PSSM data from .npz files (one per PDB structure)
- Extracts three types of information per chain:
  1. pssm_coef: Coefficient (0.0-1.0) controlling how much to trust the PSSM
  2. pssm_bias: Probability distribution over amino acids (sums to 1.0)
  3. pssm_log_odds: Log-odds ratios from PSSM (optional)

Output format: {"PDB_NAME": {"chain_A": {"pssm_coef": [...], "pssm_bias": [...], "pssm_log_odds": [...]}, ...}}
Example: {"5TTA": {"A": {"pssm_coef": [1.0, 1.0, ...], "pssm_bias": [[0.05, 0.03, ...], ...], ...}}}

Usage:
    python make_pssm_input_dict.py --jsonl_input_path parsed_pdbs.jsonl \\
                                    --PSSM_input_path /path/to/pssm/folder \\
                                    --output_path pssm_dict.jsonl

NPZ file naming convention:
    The script expects NPZ files named as: {PDB_NAME}.npz
    Each NPZ file should contain arrays named: {CHAIN}_coef, {CHAIN}_bias, {CHAIN}_odds
"""

import argparse


def main(args):
    """
    Main function to create PSSM input dictionary from NPZ files

    Args:
        args: Command line arguments containing:
            - jsonl_input_path: Path to the parsed PDB JSONL file
            - PSSM_input_path: Path to folder containing PSSM data as NPZ files
            - output_path: Path to save the PSSM dictionary in JSONL format

    NPZ file structure:
        Each NPZ file should contain the following arrays for each chain:
        - {CHAIN}_coef: [L] array of coefficients (0.0-1.0) per position
        - {CHAIN}_bias: [L, 21] array of amino acid probabilities per position
        - {CHAIN}_odds: [L, 21] array of log-odds ratios per position
    """
    import json
    import numpy as np

    # Read the parsed PDB structures from JSONL file
    with open(args.jsonl_input_path, 'r') as json_file:
        json_list = list(json_file)

    my_dict = {}

    # Process each PDB structure
    for json_str in json_list:
        result = json.loads(json_str)

        # Extract all chain IDs from the PDB structure
        all_chain_list = [
            item[-1:] for item in list(result) if item[:9] == 'seq_chain'
        ]

        # Construct path to the PSSM NPZ file for this PDB structure
        # NPZ files should be named as: {PDB_NAME}.npz
        path_to_PSSM = args.PSSM_input_path + "/" + result['name'] + ".npz"
        print(path_to_PSSM)

        # Load PSSM data from NPZ file
        pssm_input = np.load(path_to_PSSM)

        # Dictionary to store PSSM data for all chains in this structure
        pssm_dict = {}

        for chain in all_chain_list:
            # Initialize dictionary for this chain
            pssm_dict[chain] = {}

            # Extract PSSM coefficient array
            # Shape: [L] where L is the chain length
            # Values: 0.0 to 1.0
            # 0.0 = do not use PSSM at this position
            # 1.0 = fully trust PSSM at this position (ignore model predictions)
            # Intermediate values = blend model predictions with PSSM
            pssm_dict[chain]['pssm_coef'] = pssm_input[chain + '_coef'].tolist()

            # Extract PSSM bias (probability distribution) array
            # Shape: [L, 21] where L is chain length, 21 is number of amino acids
            # Each row sums to 1.0 (probability distribution over amino acids)
            # This represents the evolutionary preference at each position
            pssm_dict[chain]['pssm_bias'] = pssm_input[chain + '_bias'].tolist()

            # Extract PSSM log-odds ratios array
            # Shape: [L, 21] where L is chain length, 21 is number of amino acids
            # Log-odds scores from the PSSM (optional, not always needed)
            # Higher values indicate amino acids more likely than background
            pssm_dict[chain]['pssm_log_odds'] = pssm_input[
                chain + '_odds'].tolist()

        # Store PSSM data for this PDB structure
        my_dict[result['name']] = pssm_dict

    # Write the output dictionary to a JSONL file
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--PSSM_input_path",
                           type=str,
                           help="Path to folder containing PSSM data as NPZ files")
    argparser.add_argument(
        "--jsonl_input_path",
        type=str,
        help="Path to the parsed PDB structures in JSONL format")
    argparser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the PSSM dictionary in JSONL format")

    args = argparser.parse_args()
    main(args)

# Example output:
# {
#   "5TTA": {
#     "A": {
#       "pssm_coef": [1.0, 1.0, 0.5, ...],
#       "pssm_bias": [[0.05, 0.03, 0.02, ...], [0.04, 0.06, ...], ...],
#       "pssm_log_odds": [[1.2, -0.5, 0.3, ...], [0.8, 1.1, ...], ...]
#     },
#     "B": {...}
#   }
# }
