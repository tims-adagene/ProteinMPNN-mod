"""
ProteinMPNN Antibody Design Demo with CDR Analysis (Version 2)

This script demonstrates how to use ProteinMPNN for antibody sequence design with
a focus on Complementarity Determining Region (CDR) analysis. It can identify and
score CDR positions, highlight the top 15% most important CDR positions, and
generate visualizations of position scores.

Features:
    - Custom CDR range specification via command-line arguments
    - CDR position scoring and identification
    - Top 15% CDR position identification (highest negative log probability scores)
    - Visualization of CDR position scores with highlighted top positions
    - Support for multiple chains and homomer structures
    - Comprehensive output including scores, probabilities, and visualizations

Usage:
    # Using PDB code
    python antibody_demo_v2.py --pdb 6wgl --designed_chain C --fixed_chain "A B" \
        --cdr_ranges "C:26-35,50-66,99-114"

    # Using local PDB file
    python antibody_demo_v2.py --pdb_path ./path/to/structure.pdb \
        --designed_chain H --fixed_chain "L A" \
        --cdr_ranges "H:26-35,50-66,99-114;L:24-39,55-61,94-102"

    # For homomers
    python antibody_demo_v2.py --pdb 1234 --designed_chain "A B" --homomer

Output Files:
    - {name}_probs.npy: Amino acid probabilities for all positions
    - {name}_log_probs.npy: Log probabilities
    - {name}_scores.npy: Sample scores
    - {name}_native_score.npy: Native sequence score
    - {name}_chain_{X}_position_scores.npy: All position scores for chain X
    - {name}_chain_{X}_cdr_position_scores.npy: CDR position scores only
    - {name}_chain_{X}_top15percent_cdr_scores.npy: Top 15% CDR positions and scores
    - {name}_chain_{X}_cdr_scores.png: Visualization of CDR scores
    - {name}_chain_{X}_all_scores.npz: Complete score data for chain X

Note:
    Higher scores (negative log probabilities) indicate positions where the model
    is less confident, suggesting these positions may benefit from redesign.
"""

import os, sys
import argparse
import re
from matplotlib.offsetbox import DEBUG
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import (
    _scores_w_loss,
    _S_to_seq,
    tied_featurize,
    get_pdb,
    parse_PDB
)
from protein_mpnn_utils import (
    StructureDataset,
    StructureDatasetPDB,
    ProteinMPNN
)
from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG, VERBOSE

# Global dictionary to store custom CDR ranges
custom_cdr_ranges = {}

def parse_cdr_range(range_str):
    """
    Parse a string describing CDR ranges into a dictionary.

    CDR ranges are specified in a compact format that allows defining multiple
    ranges for multiple chains. This is useful for antibody design where you
    want to focus on specific CDR loops.

    Format: "H:26-35,50-66,99-114;L:24-39,55-61,94-102"
    Where:
        - Chains are separated by semicolons (;)
        - Each chain specification starts with chain ID followed by colon (:)
        - Ranges are separated by commas (,)
        - Each range is specified as start-end (e.g., 26-35)

    Args:
        range_str (str): String in the format described above

    Returns:
        dict: Dictionary mapping chain IDs to lists of (start, end) tuples
              Example: {'H': [(26, 35), (50, 66), (99, 114)],
                       'L': [(24, 39), (55, 61), (94, 102)]}
    """
    if not range_str:
        return {}

    result = {}
    # Split by semicolon to get each chain's specification
    chain_parts = range_str.split(';')

    for part in chain_parts:
        if ':' not in part:
            continue

        # Split chain ID from ranges
        chain, ranges = part.split(':')
        chain = chain.strip()
        result[chain] = []

        # Parse individual ranges for this chain
        for range_part in ranges.split(','):
            if '-' in range_part:
                start, end = map(int, range_part.split('-'))
                result[chain].append((start, end))

    return result

# Parse command-line arguments
parser = argparse.ArgumentParser(description='ProteinMPNN Antibody Demo with CDR Analysis')
parser.add_argument('--pdb', type=str, default=None, help='PDB code to download from RCSB')
parser.add_argument('--pdb_path', type=str, default=None, help='Path to local PDB file (overrides --pdb)')
parser.add_argument('--designed_chain', type=str, default='C', help='Chain(s) to design, comma/space separated (e.g., "C" or "H,L")')
parser.add_argument('--fixed_chain', type=str, default='A B', help='Chain(s) to keep fixed, comma/space separated (e.g., "A B")')
parser.add_argument('--cdr_ranges', type=str, default='',
                   help='Custom CDR ranges in format "H:26-35,50-66,99-114;L:24-39,55-61,94-102"')
parser.add_argument('--out_folder', type=str, default=None, help='Output folder path (default: ./outputs/temp/)')
parser.add_argument('--homomer', action='store_true', help='Treat as homomer (tie positions across chains)')

args = parser.parse_args()

# Determine PDB file path from arguments
if args.pdb_path:
    pdb_path = args.pdb_path
elif args.pdb:
    pdb = args.pdb
    pdb_path = get_pdb(pdb)
designed_chain = args.designed_chain
fixed_chain = args.fixed_chain
homomer = args.homomer

# Set output folder
if args.out_folder:
    out_folder = args.out_folder
else:
    out_folder = f'./outputs/temp/'

# Parse custom CDR ranges if provided
custom_cdr_ranges = parse_cdr_range(args.cdr_ranges)

# Create output directory
os.makedirs(out_folder, exist_ok=True)

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Model configuration
# v_48_020: Model trained with 48 edges and 0.20A noise
model_name = "v_48_020"  # Options: "v_48_002", "v_48_010", "v_48_020", "v_48_030"

# Model hyperparameters
backbone_noise = 0.00  # Standard deviation of Gaussian noise added to backbone atoms (0.00 = no noise)
hidden_dim = 128  # Hidden dimension size for neural network layers
num_layers = 3  # Number of encoder and decoder layers

# Load model checkpoint
path_to_model_weights = './vanilla_model_weights'
model_folder_path = path_to_model_weights
if model_folder_path[-1] != '/':
    model_folder_path = model_folder_path + '/'
checkpoint_path = model_folder_path + f'{model_name}.pt'

# Load and initialize model
checkpoint = torch.load(checkpoint_path, map_location=device)
print('Number of edges:', checkpoint['num_edges'])
noise_level_print = checkpoint['noise_level']
print(f'Training noise level: {noise_level_print}A')

# Initialize ProteinMPNN model
model = ProteinMPNN(num_letters=21,  # 20 amino acids + 1 unknown
                    node_features=hidden_dim,
                    edge_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_decoder_layers=num_layers,
                    augment_eps=backbone_noise,
                    k_neighbors=checkpoint['num_edges'])
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode (disables dropout, etc.)
tprint("Model loaded")

def is_in_cdr_range(chain, position, custom_ranges=None):
    """
    Check if a given position in a chain falls within any CDR range.

    CDR (Complementarity Determining Region) positions are critical for antibody
    binding and are often the focus of design efforts. This function checks whether
    a specific residue position is within any defined CDR region.

    Args:
        chain (str): Chain identifier (e.g., 'H', 'L', 'C')
        position (int): Residue position (1-indexed)
        custom_ranges (dict, optional): User-provided dictionary mapping chains to
                                       lists of (start, end) tuples. If None, returns False.

    Returns:
        bool: True if position is in any CDR range for the chain, False otherwise
    """
    # Check if custom ranges are provided and chain exists
    if custom_ranges and chain in custom_ranges:
        # Check each CDR range for this chain
        for start, end in custom_ranges[chain]:
            if start <= position <= end:
                return True

    return False

def make_tied_positions_for_homomers(pdb_dict_list):
    """
    Create tied positions dictionary for homomer structures.

    In homomer structures (multiple identical chains), we often want corresponding
    positions across all chains to have the same amino acid. This function creates
    a dictionary that ties equivalent positions across all chains together.

    Args:
        pdb_dict_list (list): List of parsed PDB dictionaries from parse_PDB

    Returns:
        dict: Dictionary mapping structure name to list of tied position dictionaries.
              Each tied position dict maps chain IDs to position lists.
              Example: {'1ABC': [{A: [1], B: [1], C: [1]}, {A: [2], B: [2], C: [2]}, ...]}
    """
    my_dict = {}
    for result in pdb_dict_list:
        # Extract all chain IDs from the PDB dictionary
        all_chain_list = sorted([
            item[-1:] for item in list(result) if item[:9] == 'seq_chain'
        ])  # Results in ['A', 'B', 'C', ...]

        tied_positions_list = []
        # Get length of first chain (all chains should have same length for homomers)
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])

        # For each position, tie it across all chains
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  # Needs to be a list for the featurize function
            tied_positions_list.append(temp_dict)

        my_dict[result['name']] = tied_positions_list
    return my_dict

# Parse chain specifications
# Convert chain strings to lists, handling various separators (commas, spaces, etc.)
if designed_chain == "":
    designed_chain_list = []
else:
    # Use regex to split by non-alphabetic characters
    designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

if fixed_chain == "":
    fixed_chain_list = []
else:
    fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

# Combine all chains that will be loaded from PDB
chain_list = list(set(designed_chain_list + fixed_chain_list))

# Design options
num_seqs = 1  # Number of sequences to generate per target
num_seq_per_target = num_seqs

# Sampling temperature for amino acid selection
# T=0.0001 is nearly deterministic (argmax), higher T increases randomness
sampling_temp = "0.0001"

# Output options
save_score = 1  # Save score=-log_prob to npy files
save_probs = 1  # Save MPNN predicted probabilities per position
score_only = 1  # Only score input backbone-sequence pairs (no design)
conditional_probs_only = 1  # Output conditional probabilities p(s_i | rest of sequence, backbone)
conditional_probs_only_backbone = 0  # If True, output p(s_i | backbone only)

# Batch processing parameters
batch_size = 1  # Batch size (can increase for GPUs with more memory)
max_length = 20000  # Maximum sequence length to process

# Additional parameters
jsonl_path = ''  # Path to folder with parsed PDB JSONLs (not used here)
omit_AAs = 'X'  # Amino acids to omit in generated sequences (X = unknown)

# PSSM (Position-Specific Scoring Matrix) parameters
pssm_multi = 0.0  # Weight for PSSM [0.0, 1.0]. 0.0 = don't use, 1.0 = ignore MPNN
pssm_threshold = 0.0  # Threshold to restrict per-position amino acids
pssm_log_odds_flag = 0  # Whether to use PSSM log odds
pssm_bias_flag = 0  # Whether to use PSSM bias

# Prepare output folder
folder_for_outputs = out_folder

# Calculate batch parameters
NUM_BATCHES = num_seq_per_target // batch_size
BATCH_COPIES = batch_size

# Parse temperature values (can specify multiple temperatures)
temperatures = [float(item) for item in sampling_temp.split()]
omit_AAs_list = omit_AAs

# Standard amino acid alphabet (20 AAs + X for unknown)
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

# Create mask for omitted amino acids
omit_AAs_np = np.array([AA in omit_AAs_list
                        for AA in alphabet]).astype(np.float32)

# Initialize various dictionaries for controlling the design
chain_id_dict = None  # Will specify which chains to design vs fix
fixed_positions_dict = None  # Can fix specific positions in specific chains
pssm_dict = None  # Position-specific scoring matrices
omit_AA_dict = None  # Per-position amino acid restrictions
bias_AA_dict = None  # Per-position amino acid biases
tied_positions_dict = None  # For tying positions across chains
bias_by_res_dict = None  # Per-residue biases
bias_AAs_np = np.zeros(len(alphabet))  # Global amino acid biases (all zeros = no bias)

# Parse PDB file and create dataset
pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
dataset_valid = StructureDatasetPDB(pdb_dict_list,
                                    truncate=None,
                                    max_length=max_length)

# Set up chain design specifications
chain_id_dict = {}
chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list,
                                           fixed_chain_list)

# Print chain information
print(chain_id_dict)
for chain in chain_list:
    l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
    print(f"Length of chain {chain} is {l}")

# Set up tied positions for homomers if requested
if homomer:
    tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
else:
    tied_positions_dict = None


# Main sequence generation and scoring loop
with torch.no_grad():
    tprint('Generating sequences...')

    for ix, protein in enumerate(dataset_valid):
        # Initialize lists to store results
        score_list = []
        loss_list = []
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []

        # Create batch copies of the protein structure
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]

        # Featurize the protein structures
        # This converts PDB data into tensors for the neural network
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
            batch_clones, device, chain_id_dict, fixed_positions_dict,
            omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)

        # Create PSSM log odds mask (1.0 for allowed, 0.0 for restricted)
        pssm_log_odds_mask = (
            pssm_log_odds_all
            > pssm_threshold).float()

        name_ = batch_clones[0]['name']

        # Score the native sequence
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx,
                          chain_encoding_all, randn_1)
        mask_for_loss = mask * chain_M * chain_M_pos
        scores, losses = _scores_w_loss(S, log_probs, mask_for_loss)
        native_score = scores.cpu().data.numpy()
        seq_loss = losses.cpu().data.numpy()

        # Generate sequences at different temperatures
        for temp in temperatures:
            for j in range(NUM_BATCHES):
                randn_2 = torch.randn(chain_M.shape, device=X.device)

                # Sample sequences (different path for tied vs untied positions)
                if tied_positions_dict == None:
                    # Standard sampling without tied positions
                    sample_dict = model.sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        bias_by_res=bias_by_res_all)
                    S_sample = sample_dict["S"]
                else:
                    # Tied sampling for homomers
                    sample_dict = model.tied_sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        tied_pos=tied_pos_list_of_lists_list[0],
                        tied_beta=tied_beta,
                        bias_by_res=bias_by_res_all)
                    S_sample = sample_dict["S"]

                # Compute scores for sampled sequences
                log_probs = model(X,
                                  S_sample,
                                  mask,
                                  chain_M * chain_M_pos,
                                  residue_idx,
                                  chain_encoding_all,
                                  randn_2,
                                  use_input_decoding_order=True,
                                  decoding_order=sample_dict["decoding_order"])
                mask_for_loss = mask * chain_M * chain_M_pos
                scores, losses = _scores_w_loss(S_sample, log_probs, mask_for_loss)

                # Move results to CPU
                scores = scores.cpu().data.numpy()
                seq_loss = losses.cpu().data.numpy()

                # Store probabilities and samples
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())

                # Process each batch copy
                for b_ix in range(BATCH_COPIES):
                    masked_chain_length_list = masked_chain_length_list_list[
                        b_ix]
                    masked_list = masked_list_list[b_ix]

                    # Calculate sequence recovery rate (how many AAs match native)
                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            torch.nn.functional.one_hot(S[b_ix], 21) *
                            torch.nn.functional.one_hot(S_sample[b_ix], 21),
                            axis=-1) * mask_for_loss[b_ix]) / torch.sum(
                                mask_for_loss[b_ix])

                    # Convert sequence indices to amino acid letters
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    score = scores[b_ix]
                    loss = seq_loss[b_ix]
                    score_list.append(score)
                    loss_list.append(loss)
                    native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])

                    # Print native sequence info (only for first sample)
                    if b_ix == 0 and j == 0 and temp == temperatures[0]:
                        start = 0
                        end = 0
                        list_of_AAs = []

                        # Split sequence by chain
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end

                        # Reorder chains to match original PDB ordering
                        native_seq = "".join(
                            list(
                                np.array(list_of_AAs)[np.argsort(
                                    masked_list)]))

                        # Add chain separators
                        l0 = 0
                        for mc_length in list(
                                np.array(masked_chain_length_list)[np.argsort(
                                    masked_list)])[:-1]:
                            l0 += mc_length
                            native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                            l0 += 1

                        # Get chain labels in correct order
                        sorted_masked_chain_letters = np.argsort(
                            masked_list_list[0])
                        print_masked_chains = [
                            masked_list_list[0][i]
                            for i in sorted_masked_chain_letters
                        ]
                        sorted_visible_chain_letters = np.argsort(
                            visible_list_list[0])
                        print_visible_chains = [
                            visible_list_list[0][i]
                            for i in sorted_visible_chain_letters
                        ]

                        # Format and print native sequence
                        native_score_print = np.format_float_positional(
                            np.float32(native_score.mean()),
                            unique=False,
                            precision=4)
                        line = '>{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n'.format(
                            name_, native_score_print, print_visible_chains,
                            print_masked_chains, model_name, native_seq)
                        print(line.rstrip())

                    # Format and print sampled sequence
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end

                    seq = "".join(
                        list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(
                                masked_list)])[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + '/' + seq[l0:]
                        l0 += 1

                    score_print = np.format_float_positional(np.float32(score),
                                                             unique=False,
                                                             precision=4)
                    seq_rec_print = np.format_float_positional(np.float32(
                        seq_recovery_rate.detach().cpu().numpy()),
                                                               unique=False,
                                                               precision=4)
                    line = '>T={}, sample={}, score={}, seq_recovery={}\n{}\n'.format(
                        temp, b_ix, score_print, seq_rec_print, seq)
                    print(line.rstrip())

# Concatenate all results
all_probs_concat = np.concatenate(all_probs_list)
all_log_probs_concat = np.concatenate(all_log_probs_list)
S_sample_concat = np.concatenate(S_sample_list)

# Save position scores and generate visualizations for each designed chain
# NOTE: Higher scores indicate positions where the model is less confident,
# suggesting these positions may benefit from redesign
for b_ix in range(BATCH_COPIES):
    # Get masked chain list and lengths
    masked_chain_length_list = masked_chain_length_list_list[b_ix]
    masked_list = masked_list_list[b_ix]

    # Process each designed chain
    for designed_chain in designed_chain_list:
        # Find index of designed chain in the masked list
        chain_idx_list = [
            i for i, chain in enumerate(masked_list) if chain == designed_chain
        ]

        if chain_idx_list:  # If chain is found
            chain_idx = chain_idx_list[0]  # Get first matching index

            # Calculate start and end positions in the concatenated sequence
            start_idx = sum(masked_chain_length_list[:chain_idx])
            end_idx = start_idx + masked_chain_length_list[chain_idx]

            # Extract scores for positions belonging to this chain
            chain_scores = seq_loss[b_ix, start_idx:end_idx]

            # Print all position scores (if VERBOSE mode enabled)
            if VERBOSE:
                tprint(f"Chain {designed_chain} all position scores:")
                for pos, score in enumerate(chain_scores):
                    actual_pos = pos + 1  # Convert to 1-indexed position
                    if mask_for_loss[b_ix, start_idx + pos] > 0:
                        print(f"Position {actual_pos}: {score:.4f}")

            # Print and collect scores for CDR regions only
            tprint(f"Chain {designed_chain} CDR position scores:")
            cdr_positions = []
            cdr_scores = []

            for pos, score in enumerate(chain_scores):
                actual_pos = pos + 1  # Convert to 1-indexed position
                # Check if this position is in a CDR region
                if mask_for_loss[b_ix, start_idx + pos] > 0 and is_in_cdr_range(designed_chain, actual_pos, custom_cdr_ranges):
                    print(f"CDR Position {actual_pos}: {score:.4f}")
                    cdr_positions.append(actual_pos)
                    cdr_scores.append(score)

            # Identify and save top 15% CDR positions (highest scores)
            if len(cdr_scores) > 0:
                # Pair scores with positions and sort by score (highest first)
                score_pos_pairs = list(zip(cdr_scores, cdr_positions))
                score_pos_pairs.sort(reverse=True)

                # Calculate number of positions in top 15%
                top_n = max(1, int(len(score_pos_pairs) * 0.15))

                tprint(f"Top {top_n} ({15:.0f}%) CDR positions with highest scores:")
                top_positions = []
                top_scores = []
                for i in range(top_n):
                    if i < len(score_pos_pairs):  # Ensure not out of range
                        score, pos = score_pos_pairs[i]
                        print(f"CDR Position {pos}: {score:.4f} (rank {i+1})")
                        top_positions.append(pos)
                        top_scores.append(score)

                # Save top 15% CDR scores to file
                # Format: first row is positions, second row is scores
                top_cdr_scores_output_file = os.path.join(
                    out_folder,
                    f"{name_}_chain_{designed_chain}_top15percent_cdr_scores.npy")
                np.save(top_cdr_scores_output_file, np.array([top_positions, top_scores]))
                print(
                    f"Chain {designed_chain} top 15% CDR position scores saved to: {top_cdr_scores_output_file}"
                )

                # Create bar plot to visualize CDR scores
                if len(cdr_positions) > 0:
                    try:
                        plt.figure(figsize=(12, 6))

                        # Create bar plot with all CDR positions
                        bars = plt.bar(range(len(cdr_positions)), cdr_scores, alpha=0.7)

                        # Configure plot
                        plt.xticks(range(len(cdr_positions)), cdr_positions, rotation=90)
                        plt.xlabel('CDR Position')
                        plt.ylabel('Score (negative log probability)')
                        plt.title(f'Chain {designed_chain} CDR Position Scores')

                        # Highlight top 15% positions in red
                        top_indices = [cdr_positions.index(pos) for pos in top_positions if pos in cdr_positions]
                        for idx in top_indices:
                            bars[idx].set_color('red')
                            bars[idx].set_alpha(1.0)

                        # Add legend
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='red', alpha=1.0, label=f'Top {top_n} positions'),
                            Patch(facecolor='blue', alpha=0.7, label='Other CDR positions')
                        ]
                        plt.legend(handles=legend_elements)

                        # Save plot
                        plot_path = os.path.join(out_folder, f"{name_}_chain_{designed_chain}_cdr_scores.png")
                        plt.tight_layout()
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"CDR scores plot saved to: {plot_path}")
                    except Exception as e:
                        print(f"Error creating plot: {str(e)}")

            # Save all CDR scores to file
            cdr_scores_output_file = os.path.join(
                out_folder,
                f"{name_}_chain_{designed_chain}_cdr_position_scores.npy")
            np.save(cdr_scores_output_file, np.array([cdr_positions, cdr_scores]))
            print(
                f"Chain {designed_chain} CDR position scores saved to: {cdr_scores_output_file}"
            )

            # Save all position scores for the chain
            chain_scores_output_file = os.path.join(
                out_folder,
                f"{name_}_chain_{designed_chain}_position_scores.npy")
            np.save(chain_scores_output_file, chain_scores)
            print(
                f"Chain {designed_chain} all position scores saved to: {chain_scores_output_file}"
            )

# Get output name from PDB
output_name = pdb_dict_list[0]['name']

tprint("Saving probabilities...")

# Save amino acid probabilities for all positions
probs_output_file = os.path.join(out_folder, f"{output_name}_probs.npy")
np.save(probs_output_file, all_probs_concat)
print(f"Probabilities saved to: {probs_output_file}")

# Save log probabilities
log_probs_output_file = os.path.join(out_folder,
                                     f"{output_name}_log_probs.npy")
np.save(log_probs_output_file, all_log_probs_concat)
print(f"Log probabilities saved to: {log_probs_output_file}")

# Save sample scores
scores_output_file = os.path.join(out_folder, f"{output_name}_scores.npy")
np.save(scores_output_file, np.array(score_list))
print(f"Sample scores saved to: {scores_output_file}")

# Save native sequence score
native_score_output_file = os.path.join(out_folder,
                                        f"{output_name}_native_score.npy")
np.save(native_score_output_file, native_score)
print(f"Native score saved to: {native_score_output_file}")

# Save complete score data for each designed chain
for designed_chain in designed_chain_list:
    chain_scores_dict = {}
    for b_ix in range(BATCH_COPIES):
        masked_chain_length_list = masked_chain_length_list_list[b_ix]
        masked_list = masked_list_list[b_ix]

        # Find chain in masked list
        chain_idx_list = [
            i for i, chain in enumerate(masked_list) if chain == designed_chain
        ]
        if chain_idx_list:
            chain_idx = chain_idx_list[0]
            # Store scores for this batch
            chain_scores_dict[f"batch_{b_ix}"] = {
                "scores":
                seq_loss[b_ix,
                         sum(masked_chain_length_list[:chain_idx]
                             ):sum(masked_chain_length_list[:chain_idx]) +
                         masked_chain_length_list[chain_idx]]
            }

    # Save to compressed numpy archive
    if chain_scores_dict:
        chain_scores_file = os.path.join(
            out_folder, f"{output_name}_chain_{designed_chain}_all_scores.npz")
        np.savez(chain_scores_file, **chain_scores_dict)
        print(
            f"Chain {designed_chain} all scores saved to: {chain_scores_file}")
