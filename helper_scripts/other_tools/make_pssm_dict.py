"""
Script to create PSSM dictionary from PSSM files for ProteinMPNN

This script converts position-specific scoring matrices (PSSMs) from text format into
a JSONL dictionary that can be used by ProteinMPNN. It reads PSSM files, converts them
to the appropriate amino acid alphabet, and packages the data for protein design.

Key features:
- Parses PSSM files in standard format (e.g., from PSI-BLAST)
- Converts amino acid alphabet from standard order to ProteinMPNN order
- Generates both probability distributions and log-odds scores
- Creates PSSM coefficients to control PSSM influence per position

The script performs alphabet permutation because PSSM files typically use the order:
ARNDCQEGHILKMFPSTWYV (standard)
while ProteinMPNN uses: ACDEFGHIKLMNPQRSTVWYX

Output format: {"PDB_NAME": {"chain_A": {"pssm_coef": [...], "pssm_bias": [...], "pssm_log_odds": [...]}, ...}}

Usage:
    Note: This is a template script with hardcoded paths. You need to modify:
    1. path_to_PSSM: Path to your PSSM file
    2. jsonl_input_path: Path to your parsed PDB JSONL file
    3. output_path: Path where to save the output dictionary
"""

import pandas as pd
import numpy as np
import glob
import random
import json


def softmax(x, T):
    """
    Compute softmax with temperature scaling

    Args:
        x: Input array
        T: Temperature parameter (controls sharpness of distribution)
           T < 1: sharper distribution
           T > 1: smoother distribution
           T = 1: standard softmax

    Returns:
        Softmax-normalized array (sums to 1.0 along last dimension)
    """
    return np.exp(x / T) / np.sum(np.exp(x / T), -1, keepdims=True)


def parse_pssm(path):
    """
    Parse a PSSM file and extract scoring matrices

    Args:
        path: Path to the PSSM file

    Returns:
        numpy array of shape [L, 40] where:
        - L is the sequence length
        - First 20 columns: log-odds scores for each amino acid
        - Last 20 columns: probability percentages for each amino acid

    PSSM file format (typical from PSI-BLAST):
        - First 2 rows are headers (skipped)
        - Each subsequent row contains position-specific scores
        - Row format: position AA [20 log-odds scores] [other data]
    """
    # Read PSSM file, skipping the first 2 header rows
    data = pd.read_csv(path, skiprows=2)

    floats_list_list = []

    # Parse each row of the PSSM
    for i in range(data.values.shape[0]):
        # Extract the numeric portion (skip first 4 characters which are position/AA)
        str1 = data.values[i][0][4:]

        # Parse all float values from this row
        floats_list = []
        for item in str1.split():
            floats_list.append(float(item))
        floats_list_list.append(floats_list)

    # Convert to numpy array
    np_lines = np.array(floats_list_list)
    return np_lines


# Parse the PSSM file
# MODIFY THIS PATH to point to your PSSM file
np_lines = parse_pssm(
    '/home/swang523/RLcage/capsid/monomersfordesign/8-16-21/pssm_rainity_final_8-16-21_int/build_0.2089_0.98_0.4653_19_2.00_0.005745.pssm'
)

# Define amino acid alphabets
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # ProteinMPNN alphabet (21 characters)
input_alphabet = 'ARNDCQEGHILKMFPSTWYV'  # Standard PSSM alphabet (20 characters)

# Create permutation matrix to convert between alphabets
# This matrix transforms scores from standard order to ProteinMPNN order
permutation_matrix = np.zeros([20, 21])
for i in range(20):
    letter1 = input_alphabet[i]
    for j in range(21):
        letter2 = mpnn_alphabet[j]
        if letter1 == letter2:
            # Mark matching positions with 1.0
            permutation_matrix[i, j] = 1.

# Extract and permute PSSM data
# First 20 columns of PSSM file contain log-odds scores
pssm_log_odds = np_lines[:, :20] @ permutation_matrix

# Last 20 columns (columns 20-39) contain probability percentages
pssm_probs = np_lines[:, 20:40] @ permutation_matrix

# Create mask for unknown amino acid (X)
# Shape: [1, 21] with zeros for known AAs and one for X
X_mask = np.concatenate([np.zeros([1, 20]), np.ones([1, 1])], -1)

# Load parsed PDBs
# MODIFY THIS PATH to point to your parsed PDB JSONL file
with open('/home/justas/projects/cages/parsed/test.jsonl', 'r') as json_file:
    json_list = list(json_file)

my_dict = {}

# Process each PDB structure
for json_str in json_list:
    result = json.loads(json_str)

    # Extract all chain IDs from the PDB structure
    all_chain_list = [
        item[-1:] for item in list(result) if item[:9] == 'seq_chain'
    ]

    pssm_dict = {}

    for chain in all_chain_list:
        pssm_dict[chain] = {}

        # PSSM coefficient: controls how much to trust PSSM vs model
        # Here set to all 1.0 (fully trust PSSM)
        # You can adjust this to values between 0.0 and 1.0 for blending
        pssm_dict[chain]['pssm_coef'] = (
            np.ones(len(result['seq_chain_A']))
        ).tolist()

        # PSSM bias: probability distribution over amino acids
        # Convert log-odds to probabilities using softmax with temperature T=1.0
        # Subtract large value (1e8) from X position to effectively exclude it
        pssm_dict[chain]['pssm_bias'] = (
            softmax(pssm_log_odds - X_mask * 1e8, 1.0)
        ).tolist()

        # PSSM log-odds: raw log-odds scores from PSSM
        pssm_dict[chain]['pssm_log_odds'] = (pssm_log_odds).tolist()

    # Store PSSM data for this PDB structure
    my_dict[result['name']] = pssm_dict

# Write output dictionary to file
# MODIFY THIS PATH to your desired output location
with open('/home/justas/projects/lab_github/mpnn/data/pssm_dict.jsonl',
          'w') as f:
    f.write(json.dumps(my_dict) + '\n')

# Example output:
# {
#   "PDB_NAME": {
#     "A": {
#       "pssm_coef": [1.0, 1.0, ...],  # L values
#       "pssm_bias": [[0.05, 0.03, ...], ...],  # [L, 21] probabilities
#       "pssm_log_odds": [[1.2, -0.5, ...], ...]  # [L, 21] log-odds
#     }
#   }
# }
