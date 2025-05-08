import os
import sys
import argparse
import numpy as np
import torch
import json
from protein_mpnn_utils import (
    tied_featurize,
    _scores_w_loss,
    _S_to_seq,
    parse_PDB
)
from protein_mpnn_utils import (
    StructureDataset, 
    StructureDatasetPDB, 
    ProteinMPNN
)
from boring_utils.utils import cprint, tprint


def highlight_changes(native_seq, designed_seq):
    """
    Add square brackets to highlight differences between native and designed sequences
    Example:
    Native seq: AABBCC, Designed seq: AADDCC
    Returns: AA[BB]CC, AA[DD]CC
    
    Args:
        native_seq: Original sequence
        designed_seq: Designed sequence
        
    Returns:
        tuple: (Annotated native sequence, Annotated designed sequence)
    """
    if len(native_seq) != len(designed_seq):
        return native_seq, designed_seq  # Skip processing if lengths differ
    
    result_native = ""
    result_designed = ""
    in_diff = False
    
    for i in range(len(native_seq)):
        if native_seq[i] != designed_seq[i]:
            if not in_diff:
                result_native += "["
                result_designed += "["
                in_diff = True
            result_native += native_seq[i]
            result_designed += designed_seq[i]
        else:
            if in_diff:
                result_native += "]"
                result_designed += "]"
                in_diff = False
            result_native += native_seq[i]
            result_designed += designed_seq[i]
    
    # Check for unclosed brackets
    if in_diff:
        result_native += "]"
        result_designed += "]"
    
    return result_native, result_designed

parser = argparse.ArgumentParser(description='ProteinMPNN Antibody Design for Top 15% CDR Positions')
parser.add_argument('--pdb_path', type=str, required=True, help='Path to PDB file')
parser.add_argument('--designed_chain', type=str, required=True, help='Chain to design, e.g. H')
parser.add_argument('--fixed_chain', type=str, default='', help='Chains to keep fixed, comma separated')
parser.add_argument('--top_cdr_file', type=str, required=True, help='NPY file with top 15% CDR positions from antibody_demo_v2.py')
parser.add_argument('--out_folder', type=str, default='./outputs/design/', help='Output folder path')
parser.add_argument('--num_seqs', type=int, default=10, help='Number of sequences to generate')
parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
parser.add_argument('--save_score', action='store_true', help='Save scores')

args = parser.parse_args()

os.makedirs(args.out_folder, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "v_48_020"
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
                    augment_eps=0.00,
                    k_neighbors=checkpoint['num_edges'])
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
tprint("Model loaded")

# Parse PDB file
designed_chain_list = args.designed_chain.split(',')
fixed_chain_list = args.fixed_chain.split(',') if args.fixed_chain else []
chain_list = list(set(designed_chain_list + fixed_chain_list))
pdb_dict_list = parse_PDB(args.pdb_path, input_chain_list=chain_list)

# Load top 15% CDR positions
top_cdr_data = np.load(args.top_cdr_file)
top_positions = top_cdr_data[0].astype(int)
tprint(f"Loaded {len(top_positions)} top CDR positions:")
cprint(top_positions)

# Create fixed_positions dictionary
fixed_positions_dict = {}
pdb_name = pdb_dict_list[0]['name']
fixed_positions_dict[pdb_name] = {}

# For each designed chain, determine which positions to fix
for chain in designed_chain_list:
    chain_seq = pdb_dict_list[0][f"seq_chain_{chain}"]
    chain_length = len(chain_seq)
    
    # Create list of fixed positions (all positions except top CDR positions)
    fixed_pos = []
    for pos in range(1, chain_length + 1):  # PDB positions start at 1
        if pos not in top_positions:
            fixed_pos.append(pos)
    
    fixed_positions_dict[pdb_name][chain] = fixed_pos

tprint(f"Created fixed positions dictionary, fixed {len(fixed_pos)} positions out of {chain_length}")

# Prepare chain_id_dict
chain_id_dict = {}
chain_id_dict[pdb_name] = (designed_chain_list, fixed_chain_list)

# Set parameters
dataset_valid = StructureDatasetPDB(pdb_dict_list,
                                    truncate=None,
                                    max_length=20000)

num_seqs = args.num_seqs
batch_size = 1
num_batches = num_seqs // batch_size
batch_copies = batch_size
temperature = args.temperature
omit_AAs_list = 'X'
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

# Run sequence generation
with torch.no_grad():
    tprint('Generating sequences with fixed positions except top 15% CDR...')
    for ix, protein in enumerate(dataset_valid):
        all_probs_list = []
        all_log_probs_list = []
        S_sample_list = []
        score_list = []
        
        # Set output file
        ali_file = os.path.join(args.out_folder, f"{pdb_name}_top_cdr_design.fa")
        with open(ali_file, 'w') as f:
            # Create batches
            batch_clones = [protein.copy() for _ in range(batch_copies)]
            
            # Featurize
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
            masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, \
            residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, \
            pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                batch_clones, device, chain_id_dict, fixed_positions_dict,
                None, None, None, None)
            
            # Calculate native score
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all, randn_1)
            mask_for_loss = mask * chain_M * chain_M_pos
            scores, losses = _scores_w_loss(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            
            # Write native sequence info
            name_ = batch_clones[0]['name']
            native_seq = _S_to_seq(S[0], chain_M[0])
            
            sorted_masked_chain_letters = np.argsort(masked_list_list[0])
            print_masked_chains = [masked_list_list[0][i] for i in sorted_masked_chain_letters]
            sorted_visible_chain_letters = np.argsort(visible_list_list[0])
            print_visible_chains = [visible_list_list[0][i] for i in sorted_visible_chain_letters]
            
            native_score_print = np.format_float_positional(np.float32(native_score.mean()), unique=False, precision=4)
            
            # Format native sequence
            start = 0
            end = 0
            list_of_AAs = []
            masked_chain_length_list = masked_chain_length_list_list[0]
            masked_list = masked_list_list[0]
            
            for mask_l in masked_chain_length_list:
                end += mask_l
                list_of_AAs.append(native_seq[start:end])
                start = end
                
            formatted_native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
            l0 = 0
            for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                l0 += mc_length
                formatted_native_seq = formatted_native_seq[:l0] + '/' + formatted_native_seq[l0:]
                l0 += 1
                
            f.write(f'>{name_}, score={native_score_print}, fixed_chains={print_visible_chains}, designed_chains={print_masked_chains}, model_name={model_name}, top_cdr_design=True\n{formatted_native_seq}\n')
            
            original_sequences = {}
            for chain_idx, chain in enumerate(print_masked_chains):
                if chain in designed_chain_list:
                    chain_start = 0
                    for i in range(chain_idx):
                        chain_start += len(list_of_AAs[i])
                    chain_end = chain_start + len(list_of_AAs[chain_idx])
                    original_sequences[chain] = formatted_native_seq[chain_start:chain_end]
            
            # Generate sequences for each batch
            for j in range(num_batches):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                
                # Generate sample
                sample_dict = model.sample(
                    X, randn_2, S, chain_M, chain_encoding_all, residue_idx,
                    mask=mask, temperature=temperature,
                    omit_AAs_np=omit_AAs_np, bias_AAs_np=np.zeros(len(alphabet)),
                    chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask,
                    pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0,
                    pssm_log_odds_flag=False, pssm_log_odds_mask=None,
                    pssm_bias_flag=False, bias_by_res=bias_by_res_all
                )
                
                S_sample = sample_dict["S"]
                
                # Calculate scores
                log_probs = model(
                    X, S_sample, mask, chain_M * chain_M_pos,
                    residue_idx, chain_encoding_all, randn_2,
                    use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"]
                )
                
                mask_for_loss = mask * chain_M * chain_M_pos
                scores, losses = _scores_w_loss(S_sample, log_probs, mask_for_loss)
                scores_np = scores.cpu().data.numpy()
                
                # Save probabilities and samples
                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                
                # Process each batch copy
                for b_ix in range(batch_copies):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    
                    # Calculate sequence recovery rate
                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            torch.nn.functional.one_hot(S[b_ix], 21) *
                            torch.nn.functional.one_hot(S_sample[b_ix], 21),
                            axis=-1
                        ) * mask_for_loss[b_ix]
                    ) / torch.sum(mask_for_loss[b_ix])
                    
                    # Get generated sequence
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    score = scores_np[b_ix]
                    score_list.append(score)
                    
                    # Format output sequence
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end
                        
                    formatted_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc_length
                        formatted_seq = formatted_seq[:l0] + '/' + formatted_seq[l0:]
                        l0 += 1
                        
                    # Highlight modified residues in the sequence
                    highlighted_seq = ""
                    current_pos = 0
                    
                    # Process each chain
                    for chain_idx, chain in enumerate(print_masked_chains):
                        chain_seq_length = len(list_of_AAs[chain_idx])
                        chain_seq = formatted_seq[current_pos:current_pos+chain_seq_length]
                        
                        # If this is a designed chain, highlight changes
                        if chain in designed_chain_list and chain in original_sequences:
                            orig_chain_seq = original_sequences[chain]
                            highlighted_chain_seq = highlight_changes(orig_chain_seq, chain_seq)[1]  # 取设计序列
                            highlighted_seq += highlighted_chain_seq
                        else:
                            highlighted_seq += chain_seq
                        
                        # Add separator if not the last chain
                        if chain_idx < len(print_masked_chains) - 1:
                            highlighted_seq += "/"
                            current_pos += chain_seq_length + 1  # +1 for the '/'
                        else:
                            current_pos += chain_seq_length
                    
                    # Print and save
                    score_print = np.format_float_positional(np.float32(score), unique=False, precision=4)
                    seq_rec_print = np.format_float_positional(np.float32(seq_recovery_rate.detach().cpu().numpy()), unique=False, precision=4)
                    sample_number = j * batch_copies + b_ix + 1
                    
                    f.write(f'>T={temperature}, sample={sample_number}, score={score_print}, seq_recovery={seq_rec_print}\n{formatted_seq}\n')
                    
                    plain_orig_seq = formatted_native_seq.replace('/', '')
                    plain_new_seq = formatted_seq.replace('/', '')
                    highlighted_orig, highlighted_new = highlight_changes(plain_orig_seq, plain_new_seq)
                    
                    print(f"Sample {sample_number}:")
                    print(f"Original: {highlighted_orig}")
                    print(f"Designed: {highlighted_new}")
                    tprint()
    
    if args.save_score:
        score_file = os.path.join(args.out_folder, f"{pdb_name}_top_cdr_design_scores.npy")
        np.save(score_file, np.array(score_list))
        print(f"Scores saved to: {score_file}")
    
    all_probs_concat = np.concatenate(all_probs_list)
    all_log_probs_concat = np.concatenate(all_log_probs_list)
    S_sample_concat = np.concatenate(S_sample_list)
    
    probs_file = os.path.join(args.out_folder, f"{pdb_name}_top_cdr_design_probs.npz")
    np.savez(probs_file, 
             probs=all_probs_concat,
             log_probs=all_log_probs_concat,
             S=S_sample_concat)
    
    print(f"Results saved to {args.out_folder}") 