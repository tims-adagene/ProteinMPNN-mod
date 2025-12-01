# -*- coding: utf-8 -*-
"""
ProteinMPNN Antibody Sequence Design Demo

This script demonstrates basic antibody sequence design using ProteinMPNN.
It scores native sequences and generates new sequences for specified antibody chains
while keeping other chains fixed. The script outputs sequence scores, probabilities,
and position-wise scores for analysis.

Features:
    - Download PDB structures from RCSB or use local files
    - Design specific chains while keeping others fixed
    - Support for homomer structures with tied positions
    - Position-wise score analysis
    - Amino acid probability predictions
    - Sequence recovery rate calculations

Usage:
    # Basic usage with hardcoded PDB
    python antibody_demo.py

    # Or modify the script to use different PDB/chains:
    # Edit pdb_path, designed_chain, and fixed_chain variables

Output Files:
    - {name}_probs.npy: Predicted amino acid probabilities
    - {name}_log_probs.npy: Log probabilities
    - {name}_scores.npy: Sample scores
    - {name}_native_score.npy: Native sequence score
    - {name}_chain_{X}_position_scores.npy: Position scores for each designed chain
    - {name}_chain_{X}_all_scores.npz: Complete score data

Note:
    This is a simplified version. For CDR-specific analysis, use antibody_demo_v2.py.
    Colab version: ./colab_notebooks/MPNN_quickdemo_score_antibody.ipynb
"""

import os, sys
import argparse
import re
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

# Input configuration
pdb = '6wgl'  # PDB code to download from RCSB
# pdb_path = get_pdb(pdb)
pdb_path = './inputs/Antibody/6wgl.pdb'  # Path to local PDB file

# Structure configuration
homomer = False  # Set to True if structure is a homomer (identical chains)
designed_chain = "C"  # Chain(s) to design (can be comma-separated like "H,L")
fixed_chain = "A B"  # Chain(s) to keep fixed (space or comma-separated)

# Set up output folder
out_folder = f'./outputs/{pdb}/'
os.makedirs(out_folder, exist_ok=True)

# Set up device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Model configuration
# v_48_020: Model trained with 48 edges and 0.20A noise
model_name = "v_48_020"  # Options: "v_48_002", "v_48_010", "v_48_020", "v_48_030"

# Model hyperparameters
backbone_noise = 0.00  # Standard deviation of Gaussian noise added to backbone atoms
hidden_dim = 128  # Hidden dimension size for neural network layers
num_layers = 3  # Number of encoder and decoder layers

# Load model weights
path_to_model_weights = './vanilla_model_weights'
model_folder_path = path_to_model_weights
if model_folder_path[-1] != '/':
    model_folder_path = model_folder_path + '/'
checkpoint_path = model_folder_path + f'{model_name}.pt'

# Load checkpoint and print info
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
model.eval()  # Set to evaluation mode
print("Model loaded")


def make_tied_positions_for_homomers(pdb_dict_list):
    """
    Create tied positions dictionary for homomer structures.

    In homomer structures (multiple identical chains), corresponding positions
    across chains can be tied to have the same amino acid. This function creates
    a dictionary that specifies which positions should be tied together.

    Args:
        pdb_dict_list (list): List of parsed PDB dictionaries from parse_PDB

    Returns:
        dict: Dictionary mapping structure name to list of tied position dictionaries.
              Each position dict maps chain IDs to position lists.
              Example: {'1ABC': [{A: [1], B: [1]}, {A: [2], B: [2]}, ...]}
    """
    my_dict = {}
    for result in pdb_dict_list:
        # Extract all chain identifiers from the PDB dictionary
        all_chain_list = sorted([
            item[-1:] for item in list(result) if item[:9] == 'seq_chain'
        ])  # Results in ['A', 'B', 'C', ...]

        tied_positions_list = []
        # Get length of first chain (all chains should have same length for homomers)
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])

        # For each position, create a dictionary tying it across all chains
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  # Needs to be a list for the featurize function
            tied_positions_list.append(temp_dict)

        my_dict[result['name']] = tied_positions_list
    return my_dict


# Parse chain specifications
# Convert chain strings to lists, handling various separators
if designed_chain == "":
    designed_chain_list = []
else:
    # Use regex to extract alphabetic characters only
    designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

if fixed_chain == "":
    fixed_chain_list = []
else:
    fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

# Combine all chains that will be processed
chain_list = list(set(designed_chain_list + fixed_chain_list))

# Design options
num_seqs = 1  # Number of sequences to generate per target
num_seq_per_target = num_seqs

# Sampling temperature
# T=0.0001 is nearly deterministic (argmax), higher values increase randomness
# Lower temperature favors high-probability amino acids
sampling_temp = "0.0001"

# Output options
save_score = 1  # Save score=-log_prob to npy files
save_probs = 1  # Save MPNN predicted probabilities per position
score_only = 1  # Only score input backbone-sequence pairs (no actual design)
conditional_probs_only = 1  # Output conditional probabilities p(s_i | rest of sequence, backbone)
conditional_probs_only_backbone = 0  # If True, output p(s_i | backbone only)

# Batch processing parameters
batch_size = 1  # Batch size (can increase for more memory)
max_length = 20000  # Maximum sequence length

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

# Parse temperature values (can specify multiple space-separated temperatures)
temperatures = [float(item) for item in sampling_temp.split()]
omit_AAs_list = omit_AAs

# Standard amino acid alphabet (20 AAs + X for unknown)
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

# Create mask for omitted amino acids
omit_AAs_np = np.array([AA in omit_AAs_list
                        for AA in alphabet]).astype(np.float32)

# Initialize control dictionaries
chain_id_dict = None  # Specifies which chains to design vs fix
fixed_positions_dict = None  # Can fix specific positions within chains
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
    print('Generating sequences...')

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
                    # S_sample contains the sampled sequence
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

                # Print scores and losses for debugging
                print(scores)
                print(losses)

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

                    # Calculate sequence recovery rate (percentage of native AAs preserved)
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

                        # Split concatenated sequence by chain
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end

                        # Reorder chains to match original PDB ordering
                        native_seq = "".join(
                            list(
                                np.array(list_of_AAs)[np.argsort(
                                    masked_list)]))

                        # Add chain separators (/)
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

                        # Format and print native sequence with metadata
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

                    # Reorder chains and add separators
                    seq = "".join(
                        list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(
                                masked_list)])[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + '/' + seq[l0:]
                        l0 += 1

                    # Format and print sample information
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

# NOTE: Position scores represent negative log probabilities.
# Higher scores indicate positions where the model is less confident,
# suggesting these positions may benefit from redesign.

# Print and save position scores for each designed chain
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

            # Print position-wise scores
            print(f"Chain {designed_chain} position scores:")
            for pos, score in enumerate(chain_scores):
                # Only print valid positions (where mask > 0)
                if mask_for_loss[b_ix, start_idx +
                                 pos] > 0:
                    # Position numbering starts at 1
                    print(f"Position {pos+1}: {score:.4f}")

            # Save scores to file
            chain_scores_output_file = os.path.join(
                out_folder,
                f"{name_}_chain_{designed_chain}_position_scores.npy")
            np.save(chain_scores_output_file, chain_scores)
            print(
                f"Chain {designed_chain} position scores saved to: {chain_scores_output_file}"
            )

# Get output name from PDB
output_name = pdb_dict_list[0]['name']

# Save amino acid probabilities
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

# Example code for loading and analyzing saved scores (commented out)
# import numpy as np
# import os
#
# file_path = "./7CR5_chain_L_position_scores.npy"
# if os.path.exists(file_path):
#     # Load the scores from the .npy file
#     position_scores = np.load(file_path)
#
#     # Print the loaded scores
#     print(f"Scores loaded from {file_path}:")
#     print(position_scores)
#
#     # Print position-wise scores in a formatted way
#     print("\nPosition-wise scores:")
#     for i, score in enumerate(position_scores):
#         print(f"Position {i+1}: {score:.4f}")
# else:
#     print(f"Error: File not found at {file_path}")
