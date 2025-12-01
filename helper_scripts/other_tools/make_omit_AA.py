"""
Script to create amino acid omission dictionary for ProteinMPNN

This script generates a dictionary that specifies which amino acids should be omitted
(not allowed) at specific residue positions during sequence design. This provides
fine-grained control over allowed amino acids at each position in each chain.

Key features:
- Position-specific amino acid restrictions
- Different restrictions for different chains
- Can specify allowed amino acids (all others are omitted)
- Useful for enforcing structural or functional constraints

Output format: {"PDB_NAME": {"chain_A": [[[positions], "omitted_AAs"], ...], "chain_B": [...]}}
Example: {"5TTA": {"A": [[[1,2,3], "GPL"], [[40,41,42,43], "WC"]], "B": []}}

The format specifies positions and the amino acids to OMIT at those positions.
For example: [[1,2,3], "GPL"] means at positions 1, 2, and 3, omit G, P, and L
(i.e., only allow the other 17 amino acids)

Usage:
    Note: This is a template script with hardcoded paths and specifications.
    You need to modify:
    1. input_path: Path to your parsed PDB JSONL file
    2. output_path: Path where to save the output dictionary
    3. The omission specifications within the script based on your design needs
"""

import glob
import random
import numpy as np
import json
import itertools

# Load parsed PDBs
# MODIFY THIS PATH to point to your parsed PDB JSONL file
with open('/home/justas/projects/lab_github/mpnn/data/pdbs.jsonl',
          'r') as json_file:
    json_list = list(json_file)

my_dict = {}

# Process each PDB structure
for json_str in json_list:
    result = json.loads(json_str)

    # Extract all chain IDs from the PDB structure
    all_chain_list = [
        item[-1:] for item in list(result) if item[:9] == 'seq_chain'
    ]

    # Dictionary to store omitted amino acids for each chain
    fixed_position_dict = {}

    print(result['name'])

    # Example specifications for specific PDB structure '5TTA'
    # MODIFY THIS SECTION based on your specific design requirements
    if result['name'] == '5TTA':
        for chain in all_chain_list:
            if chain == 'A':
                # Chain A: Define multiple groups of positions with different omissions
                fixed_position_dict[chain] = [
                    # Group 1: At positions 1-3, 7-9, 22, 25, 33, omit G, P, L
                    # (i.e., only allow the other 17 amino acids)
                    [[
                        int(item) for item in list(
                            itertools.chain(list(np.arange(
                                1, 4)), list(np.arange(7, 10)), [22, 25, 33]))
                    ], 'GPL'],

                    # Group 2: At positions 40-43, omit W and C
                    [[
                        int(item)
                        for item in list(itertools.chain([40, 41, 42, 43]))
                    ], 'WC'],

                    # Group 3: At positions 50-149, omit all except these amino acids
                    # "ACEFGHIKLMNRSTVWYX" are the OMITTED amino acids
                    # This means only D, P, Q, Y are allowed
                    [[
                        int(item) for item in list(
                            itertools.chain(list(np.arange(50, 150))))
                    ], 'ACEFGHIKLMNRSTVWYX'],

                    # Group 4: At positions 160-199, omit these amino acids
                    # This allows: A, C, D, E (the ones NOT in the omit string)
                    [[
                        int(item) for item in list(
                            itertools.chain(list(np.arange(160, 200))))
                    ], 'FGHIKLPQDMNRSTVWYX']
                ]
            else:
                # For all other chains, no amino acid restrictions
                fixed_position_dict[chain] = []
    else:
        # For all other PDB structures, no amino acid restrictions
        for chain in all_chain_list:
            fixed_position_dict[chain] = []

    # Store omission dictionary for this PDB structure
    my_dict[result['name']] = fixed_position_dict

# Write output dictionary to file
# MODIFY THIS PATH to your desired output location
with open('/home/justas/projects/lab_github/mpnn/data/omit_AA.jsonl',
          'w') as f:
    f.write(json.dumps(my_dict) + '\n')

print('Finished')

# Example output format:
# {
#   "5TTA": {
#     "A": [
#       [[1, 2, 3, 7, 8, 9, 22, 25, 33], "GPL"],
#       [[40, 41, 42, 43], "WC"],
#       [[50, 51, ..., 149], "ACEFGHIKLMNRSTVWYX"],
#       [[160, 161, ..., 199], "FGHIKLPQDMNRSTVWYX"]
#     ],
#     "B": []
#   },
#   "3LIS": {
#     "A": [],
#     "B": []
#   }
# }
#
# Interpretation:
# - At positions [1, 2, 3, 7, 8, 9, 22, 25, 33] in chain A, amino acids G, P, L are omitted
# - At positions [40, 41, 42, 43] in chain A, amino acids W, C are omitted
# - At positions [50-149] in chain A, most amino acids are omitted (only D, P, Q, Y allowed)
# - At positions [160-199] in chain A, most amino acids are omitted (only A, C, D, E allowed)
# - Chain B has no restrictions
