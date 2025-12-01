"""
Script to assign fixed and designed chains for ProteinMPNN

This script reads a JSONL file containing parsed PDB structures and creates a dictionary
that specifies which chains should be designed (redesigned) and which should be kept fixed
during the protein design process.

Output format: {"PDB_NAME": [["designed_chains"], ["fixed_chains"]]}
Example: {"5TTA": [["A"], ["B"]], "3LIS": [["A"], ["B"]]}

Usage:
    python assign_fixed_chains.py --input_path parsed_pdbs.jsonl
                                   --output_path chain_assignments.jsonl
                                   --chain_list "A B"
"""

import argparse


def main(args):
    """
    Main function to process PDB files and assign fixed/designed chains

    Args:
        args: Command line arguments containing:
            - input_path: Path to the JSONL file with parsed PDB structures
            - output_path: Path where the output dictionary will be saved
            - chain_list: Space-separated list of chains to be designed
    """
    import json

    # Read the input JSONL file containing parsed PDB data
    with open(args.input_path, 'r') as json_file:
        json_list = list(json_file)

    # Parse the global list of chains to be designed from command line arguments
    global_designed_chain_list = []
    if args.chain_list != '':
        global_designed_chain_list = [
            str(item) for item in args.chain_list.split()
        ]

    # Dictionary to store chain assignments for each PDB structure
    my_dict = {}

    # Process each PDB structure in the input file
    for json_str in json_list:
        result = json.loads(json_str)

        # Extract all chain IDs from the parsed PDB structure
        # Chain sequences are stored with keys like 'seq_chain_A', 'seq_chain_B', etc.
        all_chain_list = [
            item[-1:] for item in list(result) if item[:9] == 'seq_chain'
        ]  # Returns ['A','B', 'C',...]

        # Determine which chains to design
        if len(global_designed_chain_list) > 0:
            # Use the globally specified chain list
            designed_chain_list = global_designed_chain_list
        else:
            # If no global list provided, manually specify chains to design
            # This is a fallback/example - modify as needed
            designed_chain_list = ["A"]

        # All chains not in the designed list are kept fixed (not redesigned)
        fixed_chain_list = [
            letter for letter in all_chain_list
            if letter not in designed_chain_list
        ]

        # Store the chain assignments for this PDB structure
        my_dict[result['name']] = (designed_chain_list, fixed_chain_list)

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
                           help="Path to save the output dictionary")
    argparser.add_argument("--chain_list",
                           type=str,
                           default='',
                           help="Space-separated list of chains that need to be designed (e.g., 'A B C')")

    args = argparser.parse_args()
    main(args)

# Example output:
# {"5TTA": [["A"], ["B"]], "3LIS": [["A"], ["B"]]}
