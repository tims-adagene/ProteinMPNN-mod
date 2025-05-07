# -*- coding: utf-8 -*-
"""
colab version: ./colab_notebooks/MPNN_quickdemo_score_antibody.ipynb
"""

#@title Clone github repo
import json, time, os, sys, glob

if not os.path.isdir("ProteinMPNN"):
    os.system("git clone -q https://github.com/dauparas/ProteinMPNN.git")
sys.path.append('/content/ProteinMPNN')

#@title Setup Model
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import (loss_nll, loss_smoothed, gather_edges,
                                gather_nodes, gather_nodes_t,
                                cat_neighbors_nodes, _scores, _S_to_seq,
                                tied_featurize, get_pdb, parse_PDB)
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

#@markdown ### Input Options
pdb = '6wgl'  #@param {type:"string"}
# pdb_path = get_pdb(pdb)
pdb_path = './inputs/Antibody/6wgl.pdb'
#@markdown - pdb code (leave blank to get an upload prompt)

homomer = False  #@param {type:"boolean"}
designed_chain = "C"  #@param {type:"string"}
fixed_chain = "A B"  #@param {type:"string"}

out_folder = f'./outputs/{pdb}/'
os.makedirs(out_folder, exist_ok=True)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#v_48_010=version with 48 edges 0.10A noise
model_name = "v_48_020"  #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]

backbone_noise = 0.00  # Standard deviation of Gaussian noise added to backbone atoms
hidden_dim = 128
num_layers = 3
path_to_model_weights = './vanilla_model_weights'
model_folder_path = path_to_model_weights
if model_folder_path[-1] != '/':
    model_folder_path = model_folder_path + '/'
checkpoint_path = model_folder_path + f'{model_name}.pt'

checkpoint = torch.load(checkpoint_path, map_location=device)
print('Number of edges:', checkpoint['num_edges'])
noise_level_print = checkpoint['noise_level']
print(f'Training noise level: {noise_level_print}A')
model = ProteinMPNN(num_letters=21,
                    node_features=hidden_dim,
                    edge_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_encoder_layers=num_layers,
                    num_decoder_layers=num_layers,
                    augment_eps=backbone_noise,
                    k_neighbors=checkpoint['num_edges'])
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded")


#@title Helper functions
def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([
            item[-1:] for item in list(result) if item[:9] == 'seq_chain'
        ])  #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict


"""# To ouput Score only, modify:

```
#@markdown ### Design Options
num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
num_seq_per_target = num_seqs

#@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
sampling_temp = "0.0001" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]


save_score=1                      # 0 for False, 1 for True; save score=-log_prob to npy files
save_probs=1                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
score_only=1                      # 0 for False, 1 for True; score input backbone-sequence pairs
conditional_probs_only=1          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
```
"""

import re
import numpy as np

#########################

if designed_chain == "":
    designed_chain_list = []
else:
    designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

if fixed_chain == "":
    fixed_chain_list = []
else:
    fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

chain_list = list(set(designed_chain_list + fixed_chain_list))

#@markdown - specified which chain(s) to design and which chain(s) to keep fixed.
#@markdown   Use comma:`A,B` to specifiy more than one chain

#chain = "A" #@param {type:"string"}
#pdb_path_chains = chain
##@markdown - Define which chain to redesign

#@markdown ### Design Options
num_seqs = 1  #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
num_seq_per_target = num_seqs

#@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
sampling_temp = "0.0001"  #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]

save_score = 1  # 0 for False, 1 for True; save score=-log_prob to npy files
save_probs = 1  # 0 for False, 1 for True; save MPNN predicted probabilites per position
score_only = 1  # 0 for False, 1 for True; score input backbone-sequence pairs
conditional_probs_only = 1  # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
conditional_probs_only_backbone = 0  # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

batch_size = 1  # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
max_length = 20000  # Max sequence length

jsonl_path = ''  # Path to a folder with parsed pdb into jsonl
omit_AAs = 'X'  # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

pssm_multi = 0.0  # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
pssm_threshold = 0.0  # A value between -inf + inf to restric per position AAs
pssm_log_odds_flag = 0  # 0 for False, 1 for True
pssm_bias_flag = 0  # 0 for False, 1 for True

##############################################################
folder_for_outputs = out_folder

NUM_BATCHES = num_seq_per_target // batch_size
BATCH_COPIES = batch_size
temperatures = [float(item) for item in sampling_temp.split()]
omit_AAs_list = omit_AAs
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

omit_AAs_np = np.array([AA in omit_AAs_list
                        for AA in alphabet]).astype(np.float32)

chain_id_dict = None
fixed_positions_dict = None
pssm_dict = None
omit_AA_dict = None
bias_AA_dict = None
tied_positions_dict = None
bias_by_res_dict = None
bias_AAs_np = np.zeros(len(alphabet))

###############################################################
pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
dataset_valid = StructureDatasetPDB(pdb_dict_list,
                                    truncate=None,
                                    max_length=max_length)

chain_id_dict = {}
chain_id_dict[pdb_dict_list[0]['name']] = (designed_chain_list,
                                           fixed_chain_list)

print(chain_id_dict)
for chain in chain_list:
    l = len(pdb_dict_list[0][f"seq_chain_{chain}"])
    print(f"Length of chain {chain} is {l}")

if homomer:
    tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
else:
    tied_positions_dict = None


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores, loss


#@title RUN
with torch.no_grad():
    print('Generating sequences...')
    for ix, protein in enumerate(dataset_valid):
        score_list = []
        loss_list = []
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
            batch_clones, device, chain_id_dict, fixed_positions_dict,
            omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
        pssm_log_odds_mask = (
            pssm_log_odds_all
            > pssm_threshold).float()  #1.0 for true, 0.0 for false
        name_ = batch_clones[0]['name']

        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx,
                          chain_encoding_all, randn_1)
        mask_for_loss = mask * chain_M * chain_M_pos
        scores, losses = _scores(S, log_probs, mask_for_loss)
        native_score = scores.cpu().data.numpy()
        seq_loss = losses.cpu().data.numpy()

        for temp in temperatures:
            for j in range(NUM_BATCHES):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                if tied_positions_dict == None:
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
                    # Compute scores
                    S_sample = sample_dict["S"]
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
                scores, losses = _scores(S_sample, log_probs, mask_for_loss)
                print(scores)
                print(losses)
                scores = scores.cpu().data.numpy()
                seq_loss = losses.cpu().data.numpy()
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                for b_ix in range(BATCH_COPIES):
                    masked_chain_length_list = masked_chain_length_list_list[
                        b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            torch.nn.functional.one_hot(S[b_ix], 21) *
                            torch.nn.functional.one_hot(S_sample[b_ix], 21),
                            axis=-1) * mask_for_loss[b_ix]) / torch.sum(
                                mask_for_loss[b_ix])
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    score = scores[b_ix]
                    loss = seq_loss[b_ix]
                    score_list.append(score)
                    loss_list.append(loss)
                    native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                    if b_ix == 0 and j == 0 and temp == temperatures[0]:
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end
                        native_seq = "".join(
                            list(
                                np.array(list_of_AAs)[np.argsort(
                                    masked_list)]))
                        l0 = 0
                        for mc_length in list(
                                np.array(masked_chain_length_list)[np.argsort(
                                    masked_list)])[:-1]:
                            l0 += mc_length
                            native_seq = native_seq[:l0] + '/' + native_seq[l0:]
                            l0 += 1
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
                        native_score_print = np.format_float_positional(
                            np.float32(native_score.mean()),
                            unique=False,
                            precision=4)
                        line = '>{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n'.format(
                            name_, native_score_print, print_visible_chains,
                            print_masked_chains, model_name, native_seq)
                        print(line.rstrip())
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

all_probs_concat = np.concatenate(all_probs_list)
all_log_probs_concat = np.concatenate(all_log_probs_list)
S_sample_concat = np.concatenate(S_sample_list)

# #@markdown ### Amino acid probabilties
# import plotly.express as px
# fig = px.imshow(np.exp(all_log_probs_concat).mean(0).T,
#                 labels=dict(x="positions", y="amino acids", color="probability"),
#                 y=list(alphabet),
#                 template="simple_white"
#                )

# fig.update_xaxes(side="top")

# fig.show()

# #@markdown ### Sampling temperature adjusted amino acid probabilties
# import plotly.express as px
# fig = px.imshow(all_probs_concat.mean(0).T,
#                 labels=dict(x="positions", y="amino acids", color="probability"),
#                 y=list(alphabet),
#                 template="simple_white"
#                )

# fig.update_xaxes(side="top")

# fig.show()
"""# Save Scores

NOTE: we need to pay attention to the LARGE position_scores, they are the one that needs to be changed.
"""

# Print and save position scores for each designed chain
for b_ix in range(BATCH_COPIES):
    # Get masked chain list and lengths
    masked_chain_length_list = masked_chain_length_list_list[b_ix]
    masked_list = masked_list_list[b_ix]

    # Process each designed chain
    for designed_chain in designed_chain_list:
        # Find index of designed chain
        chain_idx_list = [
            i for i, chain in enumerate(masked_list) if chain == designed_chain
        ]

        if chain_idx_list:  # If chain is found
            chain_idx = chain_idx_list[0]  # Get first matching index

            # Calculate start and end positions in sequence
            start_idx = sum(masked_chain_length_list[:chain_idx])
            end_idx = start_idx + masked_chain_length_list[chain_idx]

            # Get scores for positions belonging to this chain
            chain_scores = seq_loss[b_ix, start_idx:end_idx]

            # Print scores
            print(f"Chain {designed_chain} position scores:")
            for pos, score in enumerate(chain_scores):
                if mask_for_loss[b_ix, start_idx +
                                 pos] > 0:  # Only print valid positions
                    print(f"Position {pos+1}: {score:.4f}")

            # Save scores to file (using chain ID in filename)
            chain_scores_output_file = os.path.join(
                out_folder,
                f"{name_}_chain_{designed_chain}_position_scores.npy")
            np.save(chain_scores_output_file, chain_scores)
            print(
                f"Chain {designed_chain} position scores saved to: {chain_scores_output_file}"
            )

output_name = pdb_dict_list[0]['name']  # Get PDB name (e.g. '7CR5')

# Save probabilities
probs_output_file = os.path.join(out_folder, f"{output_name}_probs.npy")
np.save(probs_output_file, all_probs_concat)
print(f"Probabilities saved to: {probs_output_file}")

# Save log probabilities
log_probs_output_file = os.path.join(out_folder,
                                     f"{output_name}_log_probs.npy")
np.save(log_probs_output_file, all_log_probs_concat)
print(f"Log probabilities saved to: {log_probs_output_file}")

# Save scores
scores_output_file = os.path.join(out_folder, f"{output_name}_scores.npy")
np.save(scores_output_file, np.array(score_list))
print(f"Sample scores saved to: {scores_output_file}")

# Save native score
native_score_output_file = os.path.join(out_folder,
                                        f"{output_name}_native_score.npy")
np.save(native_score_output_file, native_score)
print(f"Native score saved to: {native_score_output_file}")

# Save total scores for each designed chain
for designed_chain in designed_chain_list:
    chain_scores_dict = {}
    for b_ix in range(BATCH_COPIES):
        masked_chain_length_list = masked_chain_length_list_list[b_ix]
        masked_list = masked_list_list[b_ix]

        chain_idx_list = [
            i for i, chain in enumerate(masked_list) if chain == designed_chain
        ]
        if chain_idx_list:
            chain_idx = chain_idx_list[0]
            chain_scores_dict[f"batch_{b_ix}"] = {
                "scores":
                seq_loss[b_ix,
                         sum(masked_chain_length_list[:chain_idx]
                             ):sum(masked_chain_length_list[:chain_idx]) +
                         masked_chain_length_list[chain_idx]]
            }

    if chain_scores_dict:
        chain_scores_file = os.path.join(
            out_folder, f"{output_name}_chain_{designed_chain}_all_scores.npz")
        np.savez(chain_scores_file, **chain_scores_dict)
        print(
            f"Chain {designed_chain} all scores saved to: {chain_scores_file}")

# import numpy as np
# import os

# file_path = "./7CR5_chain_L_position_scores.npy"
# if os.path.exists(file_path):
#     # Load the scores from the .npy file
#     position_scores = np.load(file_path)

#     # Print the loaded scores
#     print(f"Scores loaded from {file_path}:")
#     print(position_scores)

#     # Optionally, print position-wise scores more nicely
#     print("\nPosition-wise scores:")
#     for i, score in enumerate(position_scores):
#         print(f"Position {i+1}: {score:.4f}")
# else:
#     print(f"Error: File not found at {file_path}")

print(all_probs_concat)

print(score_list)

print(all_log_probs_concat)

print(native_score)
