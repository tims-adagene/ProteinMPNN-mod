"""
ProteinMPNN Utilities Module

Core utilities and model classes for ProteinMPNN protein sequence design.
This module contains all the fundamental building blocks for the ProteinMPNN model.

Key Components:
1. Data Processing Functions:
   - parse_fasta: Parse FASTA sequence files
   - parse_PDB: Parse PDB structure files
   - tied_featurize: Convert protein structures to model inputs

2. Dataset Classes:
   - StructureDataset: Load protein structures from JSONL files
   - StructureDatasetPDB: Load protein structures from PDB files
   - StructureLoader: Batch loader for efficient processing

3. Model Architecture:
   - ProteinMPNN: Main sequence design model
   - ProteinFeatures: Graph-based protein structure featurization
   - CA_ProteinFeatures: CA-only structure featurization
   - EncLayer: Encoder transformer layer
   - DecLayer: Decoder transformer layer

4. Helper Functions:
   - gather_edges, gather_nodes: Graph convolution utilities
   - _scores: Calculate sequence-structure compatibility scores
   - loss_nll, loss_smoothed: Training loss functions

Model Architecture:
ProteinMPNN uses an encoder-decoder transformer architecture on protein graphs:
- Encoder: Processes backbone structure with full self-attention
- Decoder: Generates sequence autoregressively with masked self-attention
- Graph representation: k-nearest neighbor graph based on Ca distances
- Features: RBF-encoded distances, orientations, positional embeddings

Note: Some functions/classes are adopted from:
https://github.com/jingraham/neurips19-graph-protein-design
"""

from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

from boring_utils.utils import cprint, tprint
from boring_utils.helpers import DEBUG, VERBOSE


def parse_fasta(filename, limit=-1, omit=[]):
    """
    Parse a FASTA file and extract sequences.

    Args:
        filename (str): Path to FASTA file
        limit (int): Maximum number of sequences to read (-1 for all)
        omit (list): List of characters to remove from sequences (e.g., ["/"])

    Returns:
        tuple: (headers, sequences) as numpy arrays
            headers: Array of sequence names (without '>')
            sequences: Array of amino acid sequences
    """
    header = []
    sequence = []
    lines = open(filename, "r")
    for line in lines:
        line = line.rstrip()
        if line[0] == ">":
            if len(header) == limit:
                break
            header.append(line[1:])
            sequence.append([])
        else:
            if omit:
                line = [item for item in line if item not in omit]
                line = ''.join(line)
            line = ''.join(line)
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return np.array(header), np.array(sequence)


def _scores(S, log_probs, mask):
    """
    Calculate average negative log probability (score) for a sequence.
    Lower scores indicate better sequence-structure compatibility.

    Args:
        S (torch.Tensor): Amino acid sequence indices [B, L]
        log_probs (torch.Tensor): Log probabilities from model [B, L, 21]
        mask (torch.Tensor): Valid position mask [B, L] (1.0 for valid, 0.0 for padding)

    Returns:
        torch.Tensor: Average negative log probability per sequence [B]
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores


def _scores_w_loss(S, log_probs, mask):
    """
    Calculate scores and per-position losses.

    Args:
        S (torch.Tensor): Amino acid sequence indices [B, L]
        log_probs (torch.Tensor): Log probabilities from model [B, L, 21]
        mask (torch.Tensor): Valid position mask [B, L]

    Returns:
        tuple: (scores, loss)
            scores: Average negative log probability [B]
            loss: Per-position negative log probability [B, L]
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores, loss


def _S_to_seq(S, mask):
    """
    Convert amino acid indices to sequence string.

    Args:
        S (torch.Tensor): Amino acid indices [L]
        mask (torch.Tensor): Valid position mask [L]

    Returns:
        str: Amino acid sequence (only valid positions)
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join(
        [alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
    return seq


def get_pdb(pdb_code="", local_path=None):
    """Fetch PDB file
    
    Args:
        pdb_code (str): PDB code, if provided will download from RCSB PDB
        local_path (str): Local PDB file path, if provided will use local file
        
    Returns:
        str: Path to PDB file
    """
    if local_path and os.path.exists(local_path):
        return local_path
    elif pdb_code and pdb_code.strip():
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
    else:
        raise ValueError(
            "Must provide either local PDB file path or valid PDB code")


def parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    """
    Parse PDB file and extract backbone atom coordinates and sequence.
    Used internally by parse_PDB function.

    This function reads a PDB file and extracts coordinates for specified atoms
    along with the amino acid sequence. It handles insertion codes and
    converts MSE (selenomethionine) to MET.

    Args:
        x (str): PDB filename/path
        atoms (list): List of atom types to extract, default ['N', 'CA', 'C']
        chain (str): Specific chain ID to parse, or None for first chain

    Returns:
        tuple: (coordinates, sequence) or ('no_chain', 'no_chain') if parsing fails
            coordinates: numpy array of shape [L, num_atoms, 3]
            sequence: numpy array of single-letter amino acid codes

    PDB file structure:
        ch: chain ID
        atom: atom type (N, CA, C, O, etc.)
        resi: residue name (3-letter code)
        resn: residue number
        resa: residue insertion code (if any)
        x, y, z: atom coordinates

    Internal data structures:
        xyz: nested dict xyz[residue_number][insertion_code][atom_name] = [x, y, z]
        seq: nested dict seq[residue_number][insertion_code] = three_letter_code
    """

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'GAP'
    ]

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x)
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x)
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1

                # resn = int(resn)
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)
            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))
        return (
            np.array(xyz_).reshape(-1, len(atoms), 3), 
            N_to_AA(np.array(seq_))
        )
    except TypeError:
        return 'no_chain', 'no_chain'


def parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False):
    """
    Parse PDB file and extract all chain structures into a dictionary format.

    This is the main PDB parsing function that processes multi-chain protein
    structures and organizes the data for use with ProteinMPNN.

    Args:
        path_to_pdb (str): Path to PDB file
        input_chain_list (list): Specific chains to parse, or None for all chains
        ca_only (bool): If True, only extract CA atoms; if False, extract N,CA,C,O

    Returns:
        list: List of dictionaries, each containing:
            - 'name': PDB filename without extension
            - 'num_of_chains': Number of chains parsed
            - 'seq': Concatenated sequence of all chains
            - 'seq_chain_X': Sequence for chain X
            - 'coords_chain_X': Dictionary of coordinates for chain X
                - If ca_only: {'CA_chain_X': [[x,y,z], ...]}
                - Otherwise: {'N_chain_X': coords, 'CA_chain_X': coords,
                             'C_chain_X': coords, 'O_chain_X': coords}

    Chain naming:
        Uses uppercase (A-Z), lowercase (a-z), then numbers (0-299)
        to support up to 352 chains
    """
    c = 0
    pdb_dict_list = []

    # Create chain alphabet: A-Z, a-z, 0-299
    # Supports up to 352 chains in a single structure
    init_alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        # concat_N = []
        # concat_CA = []
        # concat_C = []
        # concat_O = []
        # concat_mask = []
        # coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']

            xyz, seq = parse_PDB_biounits(biounit,
                                          atoms=sidechain_atoms,
                                          chain=letter)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                else:
                    coords_dict_chain['N_chain_' +
                                      letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' +
                                      letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' +
                                      letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' +
                                      letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def tied_featurize(batch,
                   device,
                   chain_dict,
                   fixed_position_dict=None,
                   omit_AA_dict=None,
                   tied_positions_dict=None,
                   pssm_dict=None,
                   bias_by_res_dict=None,
                   ca_only=False):
    """
    Convert protein structures into model input tensors with constraints.

    This is the core featurization function that processes protein structures
    and prepares them for the ProteinMPNN model. It handles multi-chain proteins,
    applies design constraints, and creates properly formatted tensors.

    NOTE: Despite the name, this function primarily packs and pads data into
    torch tensors rather than performing feature extraction (which happens in
    the ProteinFeatures module).

    Args:
        batch (list): List of protein structure dictionaries from parse_PDB
        device (torch.device): CUDA or CPU device
        chain_dict (dict): Specifies which chains to design vs. keep fixed
            Format: {pdb_name: ([designed_chains], [fixed_chains])}
            Example: {"1ABC": (["A", "B"], ["C"])}
        fixed_position_dict (dict): Positions to keep unchanged
            Format: {pdb_name: {chain: [position_indices]}}
        omit_AA_dict (dict): Amino acids to prohibit at specific positions
            Format: {pdb_name: {chain: [([positions], [amino_acids])]}}
        tied_positions_dict (dict): Positions that must have the same amino acid
            Critical for symmetric proteins/homooligomers
            Example for homomer: tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
        pssm_dict (dict): Position-specific scoring matrices to guide design
            Contains 'pssm_coef', 'pssm_bias', 'pssm_log_odds' per position
        bias_by_res_dict (dict): Per-residue amino acid biases
        ca_only (bool): If True, use CA-only coordinates

    Returns:
        tuple: (X, S, mask, lengths, chain_M, chain_encoding_all, ...)
            X: Backbone coordinates [B, L, 4, 3] or [B, L, 3] if ca_only
            S: Amino acid sequences (integer encoded) [B, L]
            mask: Valid residue mask [B, L]
            lengths: Sequence lengths per sample
            chain_M: Design mask (1=design, 0=fixed) [B, L]
            chain_encoding_all: Chain IDs [B, L]
            ... (additional constraint and metadata tensors)

    Processing steps:
        1. Initialize coordinate and mask tensors
        2. Separate visible (fixed) and masked (designable) chains
        3. Build per-chain coordinates and masks
        4. Apply fixed position constraints
        5. Apply tied position constraints (for symmetry)
        6. Concatenate chains and pad to maximum length
        7. Convert to torch tensors on specified device
    """
    if DEBUG or VERBOSE: 
        cprint(batch[0].keys())
        cprint(chain_dict)

    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch],
                       dtype=np.int32)  #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])

    # Step 1: Init X atom coordinates (B, L_max, ?, 3)
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros(
        [B, L_max],
        dtype=np.int32)  #1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros(
        [B, L_max],
        dtype=np.float32)  #1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros(
        [B, L_max, 21],
        dtype=np.float32)  #1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones(
        [B, L_max, 21],
        dtype=np.float32)  #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros(
        [B, L_max],
        dtype=np.int32)  #1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros(
        [B, L_max],
        dtype=np.int32)  #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)

    # Step 2: Distinguish visible chains and masked chains
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[b[
                'name']]  #masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [
                item[-1:] for item in list(b) if item[:10] == 'seq_chain_'
            ]
            visible_chains = []
        masked_chains.sort()  #sort masked_chains
        visible_chains.sort()  #sort visible_chains
        all_chains = masked_chains + visible_chains
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0

        # Step 3: Build the batch
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join(
                    [a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] +
                                             chain_length)
                chain_coords = b[
                    f'coords_chain_{letter}']  #this is a dictionary
                chain_mask = np.zeros(chain_length)  #0.0 for visible chains
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']
                                       )  #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([
                        chain_coords[c] for c in [
                            f'N_chain_{letter}', f'CA_chain_{letter}',
                            f'C_chain_{letter}', f'O_chain_{letter}'
                        ]
                    ], 1)  #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(
                    c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros(
                    [chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join(
                    [a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] +
                                             chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[
                    f'coords_chain_{letter}']  #this is a dictionary
                chain_mask = np.ones(chain_length)  #1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}']
                                       )  #[chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([
                        chain_coords[c] for c in [
                            f'N_chain_{letter}', f'CA_chain_{letter}',
                            f'C_chain_{letter}', f'O_chain_{letter}'
                        ]
                    ], 1)  #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(
                    c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)

                # fixed position restraint
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros(
                    [chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array([
                            np.argwhere(np.array(list(alphabet)) == AA)[0][0]
                            for AA in item[1]
                        ]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA
                                         for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[
                            b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(
                        bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        # Step 4: Tied position restraint
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(
                    list(
                        itertools.chain(
                            *[list(item) for item in tied_pos_list])))

                # NOTE:
                # Enforce same/related amino acids at specified positions
                # tied_beta: coefficient controlling association strength per position  
                # Critical for designing symmetric or repetitive protein structures
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(
                            letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx + v[0][v_count] -
                                                1)  #make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] -
                                          1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ -
                                                1)  #make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        # Step 5: Concatenate all chains
        x = np.concatenate(x_chain_list, 0)  #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,
                           0)  #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(
            fixed_position_mask_list,
            0)  #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(
            pssm_coef_list, 0)  #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(
            pssm_bias_list, 0)  #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(
            pssm_log_odds_list,
            0)  #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(
            bias_by_res_list, 0
        )  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]],
                       'constant',
                       constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]],
                       'constant',
                       constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]],
                           'constant',
                           constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list, 0),
                                  [[0, L_max - l]],
                                  'constant',
                                  constant_values=(0.0, ))
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[
            i,
        ] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]],
                                    'constant',
                                    constant_values=(0.0, ))
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max - l]],
                               'constant',
                               constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max - l], [0, 0]],
                               'constant',
                               constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, L_max - l], [0, 0]],
                                   'constant',
                                   constant_values=(0.0, ))

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0, L_max - l], [0, 0]],
                                 'constant',
                                 constant_values=(0.0, ))
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence],
                             dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32,
                                                       device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32,
                                                       device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(
        dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32,
                                               device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(
        np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32,
                                                           device=device)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate(
        [phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]],
        -1)  #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32,
                                                       device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,
                                                   device=device)

    # Step 6: Convert to torch tensors
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32,
                                                   device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32,
                                                     device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X
    
    if DEBUG:
        cprint(X_out.shape)
        cprint(S.shape)
        cprint(chain_M.shape)

    return (X_out, S, mask, lengths, chain_M, chain_encoding_all, letter_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta)


def loss_nll(S, log_probs, mask):
    """
    Calculate negative log-likelihood loss for sequence design.

    Standard cross-entropy loss used during ProteinMPNN training.

    Args:
        S (torch.Tensor): Ground truth amino acid indices [B, L]
        log_probs (torch.Tensor): Predicted log probabilities [B, L, 21]
        mask (torch.Tensor): Valid position mask [B, L]

    Returns:
        tuple: (loss, loss_av)
            loss: Per-position negative log likelihood [B, L]
            loss_av: Average loss across all valid positions (scalar)
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """
    Calculate label-smoothed cross-entropy loss.

    Label smoothing prevents overconfidence and can improve generalization.
    Instead of hard targets (one-hot), uses soft targets with small probability
    mass distributed across all classes.

    Args:
        S (torch.Tensor): Ground truth amino acid indices [B, L]
        log_probs (torch.Tensor): Predicted log probabilities [B, L, 21]
        mask (torch.Tensor): Valid position mask [B, L]
        weight (float): Smoothing weight (default 0.1)
            Higher values = more smoothing

    Returns:
        tuple: (loss, loss_av)
            loss: Per-position smoothed loss [B, L]
            loss_av: Average loss across all valid positions (scalar)
    """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Apply label smoothing: add small probability to all classes
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


class StructureDataset():
    """
    Dataset for loading protein structures from JSONL files.

    Each line in the JSONL file should be a JSON object containing:
    - 'name': Protein name/identifier
    - 'seq': Amino acid sequence
    - 'coords': Dictionary of backbone atom coordinates per chain

    Args:
        jsonl_file (str): Path to JSONL file
        verbose (bool): Print loading progress
        truncate (int): Limit number of structures to load (None for all)
        max_length (int): Maximum sequence length to include
        alphabet (str): Valid amino acid characters

    Attributes:
        data (list): List of protein structure dictionaries
    """

    def __init__(self,
                 jsonl_file,
                 verbose=True,
                 truncate=None,
                 max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {'bad_chars': 0, 'too_long': 0, 'bad_seq_length': 0}

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Convert raw coords to np arrays
                #for key, val in entry['coords'].items():
                #    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        if True:
                            self.data.append(entry)
                        else:
                            discard_count['bad_seq_length'] += 1
                    else:
                        discard_count['too_long'] += 1
                else:
                    if verbose:
                        print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(
                        len(self.data), i + 1, elapsed))
            if verbose:
                print('discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB():
    """
    Dataset for loading protein structures from parsed PDB dictionaries.

    Similar to StructureDataset but takes pre-parsed PDB dictionaries
    instead of reading from JSONL files.

    Args:
        pdb_dict_list (list): List of dictionaries from parse_PDB()
        verbose (bool): Print loading progress
        truncate (int): Limit number of structures to load (None for all)
        max_length (int): Maximum sequence length to include
        alphabet (str): Valid amino acid characters

    Attributes:
        data (list): List of protein structure dictionaries
    """

    def __init__(self,
                 pdb_dict_list,
                 verbose=True,
                 truncate=None,
                 max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {'bad_chars': 0, 'too_long': 0, 'bad_seq_length': 0}

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    """
    Custom data loader for protein structures with smart batching.

    Clusters proteins by sequence length to minimize padding and maximize
    GPU utilization. Unlike standard DataLoader, this creates batches where
    total_sequence_length * num_proteins <= batch_size.

    Args:
        dataset: StructureDataset or StructureDatasetPDB instance
        batch_size (int): Maximum total sequence length per batch
        shuffle (bool): Shuffle batch order (not individual samples)
        collate_fn: Function to collate batch (default: identity function)
        drop_last (bool): Drop incomplete final batch

    Attributes:
        clusters (list): List of batches, each containing structure indices
    """

    def __init__(self,
                 dataset,
                 batch_size=100,
                 shuffle=True,
                 collate_fn=lambda x: x,
                 drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


# Graph neural network utility functions
def gather_edges(edges, neighbor_idx):
    """
    Gather edge features for neighboring nodes.

    Used in graph convolutions to collect edge features connecting to neighbors.

    Args:
        edges (torch.Tensor): Edge features [B, N, N, C]
            Full pairwise edge feature matrix
        neighbor_idx (torch.Tensor): Neighbor indices [B, N, K]
            K nearest neighbors for each node

    Returns:
        torch.Tensor: Neighbor edge features [B, N, K, C]
    """
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """
    Gather node features for neighboring nodes.

    Collects features from K nearest neighbors for each node.
    Essential for message passing in graph neural networks.

    Args:
        nodes (torch.Tensor): Node features [B, N, C]
        neighbor_idx (torch.Tensor): Neighbor indices [B, N, K]

    Returns:
        torch.Tensor: Neighbor node features [B, N, K, C]

    Implementation:
        1. Flatten neighbor indices: [B, N, K] -> [B, N*K]
        2. Gather features: [B, N, C] -> [B, N*K, C]
        3. Reshape: [B, N*K, C] -> [B, N, K, C]
    """
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    """
    Gather node features at specific indices (time-step version).

    Similar to gather_nodes but for selecting specific nodes rather than
    neighborhoods. Used during autoregressive sequence generation.

    Args:
        nodes (torch.Tensor): Node features [B, N, C]
        neighbor_idx (torch.Tensor): Node indices [B, K]

    Returns:
        torch.Tensor: Selected node features [B, K, C]
    """
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """
    Concatenate node features with their neighbors' features.

    Combines central node and neighbor information for message passing.

    Args:
        h_nodes (torch.Tensor): Node features [B, N, C]
        h_neighbors (torch.Tensor): Edge/neighbor features [B, N, K, C']
        E_idx (torch.Tensor): Neighbor indices [B, N, K]

    Returns:
        torch.Tensor: Concatenated features [B, N, K, C+C']
    """
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    """
    Encoder layer for ProteinMPNN.

    Performs message passing on the protein graph with full self-attention.
    Updates both node (residue) and edge (pairwise) features through
    geometric attention mechanism.

    Architecture:
        1. Node update: Aggregate messages from neighbors
        2. Feed-forward network on nodes
        3. Edge update: Update pairwise features

    Args:
        num_hidden (int): Hidden dimension size
        num_in (int): Input feature dimension
        dropout (float): Dropout probability
        num_heads: Not used (kept for compatibility)
        scale (int): Normalization scale for aggregation (default 30)
    """

    def __init__(self,
                 num_hidden,
                 num_in,
                 dropout=0.1,
                 num_heads=None,
                 scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """
        Forward pass: update node and edge features through message passing.

        Args:
            h_V (torch.Tensor): Node features [B, N, C]
            h_E (torch.Tensor): Edge features [B, N, K, C]
            E_idx (torch.Tensor): Neighbor indices [B, N, K]
            mask_V (torch.Tensor): Node mask [B, N]
            mask_attend (torch.Tensor): Attention mask [B, N, K]

        Returns:
            tuple: (h_V, h_E) updated node and edge features
        """
        # Info aggregation and message passing
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # Mask filtering
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        # Mask filtering
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    """
    Decoder layer for ProteinMPNN.

    Performs autoregressive sequence generation with masked self-attention.
    Only allows attention to previously generated positions and encoder output.

    Architecture:
        1. Masked message passing (only attend to "past" positions)
        2. Feed-forward network on nodes

    Args:
        num_hidden (int): Hidden dimension size
        num_in (int): Input feature dimension
        dropout (float): Dropout probability
        num_heads: Not used (kept for compatibility)
        scale (int): Normalization scale for aggregation (default 30)
    """

    def __init__(self,
                 num_hidden,
                 num_in,
                 dropout=0.1,
                 num_heads=None,
                 scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """
        Forward pass: update node features with masked attention.

        Args:
            h_V (torch.Tensor): Node features [B, N, C]
            h_E (torch.Tensor): Combined edge features [B, N, K, C]
            mask_V (torch.Tensor): Node mask [B, N]
            mask_attend (torch.Tensor): Attention mask [B, N, K]
                Controls which positions can attend to which

        Returns:
            torch.Tensor: Updated node features [B, N, C]
        """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP applied independently to each position.
    Standard component of transformer architectures.

    Args:
        num_hidden (int): Input/output dimension
        num_ff (int): Hidden layer dimension (usually 4x num_hidden)
    """

    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        """Apply feed-forward network."""
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    """
    Relative positional encodings for protein sequences.

    Encodes relative position between residues along the sequence.
    Uses learnable embeddings for position differences within a window.

    Args:
        num_embeddings (int): Embedding dimension
        max_relative_feature (int): Maximum relative distance to encode
            Positions beyond this use a special token
    """

    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1,
                                num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask + (1 - mask) * (
                           2 * self.max_relative_feature + 1)
        d_onehot = torch.nn.functional.one_hot(
            d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class CA_ProteinFeatures(nn.Module):
    """
    Extract graph features from CA-only protein structures.

    Simplified version of ProteinFeatures that only uses CA (alpha carbon) atoms.
    Useful for low-resolution structures or when full backbone is unavailable.

    Creates k-nearest neighbor graph based on CA-CA distances and encodes:
    - Distance features using radial basis functions (RBF)
    - Orientation features (backbone geometry)
    - Positional encodings (sequence separation)

    Args:
        edge_features (int): Edge feature dimension
        node_features (int): Node feature dimension (not used in CA-only)
        num_positional_embeddings (int): Dimension for positional encodings
        num_rbf (int): Number of RBF kernels for distance encoding
        top_k (int): Number of nearest neighbors
        augment_eps (float): Gaussian noise level for coordinate augmentation
        num_chain_embeddings (int): Not used (kept for compatibility)
    """

    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 augment_eps=0.,
                 num_chain_embeddings=16):
        super(CA_ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        # Positional encoding
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        # Normalization and embedding
        node_in, edge_in = 3, num_positional_embeddings + num_rbf * 9 + 7
        self.node_embedding = nn.Linear(node_in, node_features,
                                        bias=False)  #NOT USED
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_nodes = nn.LayerNorm(node_features)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(
            torch.abs(1 + torch.stack(
                [Rxx - Ryy - Rzz, -Rxx + Ryy - Rzz, -Rxx - Ryy + Rzz], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(
            torch.stack([
                _R(2, 1) - _R(1, 2),
                _R(0, 2) - _R(2, 0),
                _R(1, 0) - _R(0, 1)
            ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def _orientations_coarse(self, X, E_idx, eps=1e-6):
        dX = X[:, 1:, :] - X[:, :-1, :]
        dX_norm = torch.norm(dX, dim=-1)
        dX_mask = (3.6 < dX_norm) & (dX_norm < 4.0)  #exclude CA-CA jumps
        dX = dX * dX_mask[:, :, None]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D),
                                   torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)
        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self._quaternions(R)

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)
        return AD_features, O_features

    def _dist(self, X, mask, eps=1E-6):
        """ Pairwise euclidean distances """
        # Convolutional network on NCHW
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        # Identify k nearest neighbors (including self)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust,
                                        np.minimum(self.top_k, X.shape[1]),
                                        dim=-1,
                                        largest=False)
        mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
        return D_neighbors, E_idx, mask_neighbors

    def _rbf(self, D):
        # Distance radial basis function
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, -1) +
            1e-6)  #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None],
                                       E_idx)[:, :, :, 0]  #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, Ca, mask, residue_idx, chain_labels):
        """ Featurize coordinates as an attributed graph """
        if self.augment_eps > 0:
            Ca = Ca + self.augment_eps * torch.randn_like(Ca)

        D_neighbors, E_idx, mask_neighbors = self._dist(Ca, mask)

        Ca_0 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_2 = torch.zeros(Ca.shape, device=Ca.device)
        Ca_0[:, 1:, :] = Ca[:, :-1, :]
        Ca_1 = Ca
        Ca_2[:, :-1, :] = Ca[:, 1:, :]

        V, O_features = self._orientations_coarse(Ca, E_idx)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  #Ca_1-Ca_1
        RBF_all.append(self._get_rbf(Ca_0, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_0, Ca_1, E_idx))
        RBF_all.append(self._get_rbf(Ca_0, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_1, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_1, Ca_2, E_idx))

        RBF_all.append(self._get_rbf(Ca_2, Ca_0, E_idx))
        RBF_all.append(self._get_rbf(Ca_2, Ca_1, E_idx))

        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :,
                                                            0]  #[B, L, K]

        d_chains = ((chain_labels[:, :, None] -
                     chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all, O_features), -1)

        E = self.edge_embedding(E)
        E = self.norm_edges(E)

        return E, E_idx


class ProteinFeatures(nn.Module):
    """
    Extract graph features from full-backbone protein structures.

    Full version using N, CA, C, O backbone atoms for richer geometric features.
    Creates k-nearest neighbor graph and encodes detailed structural information.

    Features computed:
    - RBF-encoded distances for 25 atom pairs (all combinations of N,CA,C,O,Cb)
    - Virtual Cb positions (computed geometrically from N,CA,C)
    - Positional encodings (sequence separation)
    - Chain encodings (intra vs. inter-chain interactions)

    Args:
        edge_features (int): Edge feature dimension
        node_features (int): Node feature dimension (not used)
        num_positional_embeddings (int): Dimension for positional encodings
        num_rbf (int): Number of RBF kernels for distance encoding
        top_k (int): Number of nearest neighbors in graph
        augment_eps (float): Gaussian noise level for coordinate augmentation
            Used during training for robustness
        num_chain_embeddings (int): Not used (kept for compatibility)
    """

    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 augment_eps=0.,
                 num_chain_embeddings=16):
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust,
                                        np.minimum(self.top_k, X.shape[1]),
                                        dim=-1,
                                        largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(
            torch.sum(
                (A[:, :, None, :] - B[:, None, :, :]) ** 2, -1
            ) + 1e-6)  #[B, L, L]
        D_A_B_neighbors = gather_edges(
                            D_A_B[:, :, :, None],
                            E_idx
                        )[:, :, :, 0]  #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        """
        X: [B, L, 4, 3] : [batch, seq_len, atom type, xyz]
        mask: [B, L] : [batch, seq_len]
        residue_idx: [B, L] : [batch, seq_len]
        chain_labels: [B, L] : [batch, seq_len]

        checkout parse_PDB_biounits
        """
        if self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        b = X[:, :, 1, :] - X[:, :, 0, :]
        c = X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        Ca = X[:, :, 1, :]
        N = X[:, :, 0, :]
        C = X[:, :, 2, :]
        O = X[:, :, 3, :]

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))  #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx))  #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx))  #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx))  #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx))  #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx))  #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx))  #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx))  #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx))  #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx))  #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx))  #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx))  #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx))  #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx))  #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx))  #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx))  #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx))  #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx))  #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx))  #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx))  #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx))  #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx))  #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx))  #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx))  #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx))  #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :,
                                                            0]  #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
                    ).long()  #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


class ProteinMPNN(nn.Module):
    """
    ProteinMPNN: Structure-based protein sequence design model.

    Encoder-decoder transformer architecture that designs amino acid sequences
    for given protein backbone structures. Uses graph neural networks on
    k-nearest neighbor graphs of protein residues.

    Architecture Overview:
    1. Feature Extraction: Convert backbone coordinates to graph features
    2. Encoder: Process structure with full self-attention (3 layers)
    3. Decoder: Generate sequence autoregressively with masked attention (3 layers)
    4. Output: Per-position amino acid probability distributions

    Key Capabilities:
    - Fixed-backbone sequence design
    - Multi-chain protein design
    - Conditional design (keep some positions/chains fixed)
    - Tied positions (for symmetric designs)
    - PSSM-guided design
    - Temperature-based diversity control

    Args:
        num_letters (int): Vocabulary size (21 for 20 AAs + unknown)
        node_features (int): Node feature dimension
        edge_features (int): Edge feature dimension
        hidden_dim (int): Hidden dimension for transformer layers
        num_encoder_layers (int): Number of encoder layers (default 3)
        num_decoder_layers (int): Number of decoder layers (default 3)
        vocab (int): Vocabulary size (same as num_letters)
        k_neighbors (int): Number of nearest neighbors in graph (default 64)
        augment_eps (float): Coordinate noise for augmentation (default 0.05)
        dropout (float): Dropout probability (default 0.1)
        ca_only (bool): Use CA-only features vs. full backbone

    Attributes:
        features: ProteinFeatures or CA_ProteinFeatures module
        W_e: Edge embedding layer
        W_s: Sequence embedding layer
        encoder_layers: List of encoder transformer layers
        decoder_layers: List of decoder transformer layers
        W_out: Output projection to vocabulary
    """

    def __init__(self,
                 num_letters,
                 node_features,
                 edge_features,
                 hidden_dim,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 vocab=21,
                 k_neighbors=64,
                 augment_eps=0.05,
                 dropout=0.1,
                 ca_only=False):
        super(ProteinMPNN, self).__init__()

        # Model hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        if ca_only:
            self.features = CA_ProteinFeatures(node_features,
                                               edge_features,
                                               top_k=k_neighbors,
                                               augment_eps=augment_eps)
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(node_features,
                                            edge_features,
                                            top_k=k_neighbors,
                                            augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding_all,
                randn,
                use_input_decoding_order=False,
                decoding_order=None):
        """
        Forward pass: compute log probabilities for a sequence given structure.

        Used for scoring existing sequences or computing gradients during training.

        Args:
            X (torch.Tensor): Backbone coordinates [B, L, 4, 3] or [B, L, 3] if CA-only
            S (torch.Tensor): Amino acid sequence indices [B, L]
            mask (torch.Tensor): Valid residue mask [B, L]
            chain_M (torch.Tensor): Design mask (1=design, 0=fixed) [B, L]
            residue_idx (torch.Tensor): Residue indices for positional encoding [B, L]
            chain_encoding_all (torch.Tensor): Chain IDs [B, L]
            randn (torch.Tensor): Random noise for decoding order [B, L]
            use_input_decoding_order (bool): Use provided decoding order instead of random
            decoding_order (torch.Tensor): Explicit decoding order if provided [B, L]

        Returns:
            torch.Tensor: Log probabilities [B, L, 21]

        Data Flow:
            1. Extract graph features from structure (E, E_idx)
            2. Encode structure with full self-attention
            3. Decode sequence autoregressively with masked attention
            4. Project to amino acid logits and apply log softmax
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]),
                          device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  #update chain_M to include missing regions
        if not use_input_decoding_order:
            decoding_order = torch.argsort(
                (chain_M + 0.0001) * (torch.abs(randn))
            )  #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see.
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(self,
               X,
               randn,
               S_true,
               chain_mask,
               chain_encoding_all,
               residue_idx,
               mask=None,
               temperature=1.0,
               omit_AAs_np=None,
               bias_AAs_np=None,
               chain_M_pos=None,
               omit_AA_mask=None,
               pssm_coef=None,
               pssm_bias=None,
               pssm_multi=None,
               pssm_log_odds_flag=None,
               pssm_log_odds_mask=None,
               pssm_bias_flag=None,
               bias_by_res=None):
        """
        Sample amino acid sequences for a given protein backbone.

        Autoregressive sampling with various constraints and biases.
        Positions are sampled in random order (controlled by randn).

        Args:
            X (torch.Tensor): Backbone coordinates [B, L, 4, 3] or [B, L, 3]
            randn (torch.Tensor): Random noise for decoding order [B, L]
            S_true (torch.Tensor): Ground truth sequence (for fixed positions) [B, L]
            chain_mask (torch.Tensor): Positions to design (1) vs. keep fixed (0) [B, L]
            chain_encoding_all (torch.Tensor): Chain IDs [B, L]
            residue_idx (torch.Tensor): Residue indices [B, L]
            mask (torch.Tensor): Valid residue mask [B, L]
            temperature (float): Sampling temperature (higher = more diverse)
            omit_AAs_np (np.array): Global amino acid omission mask [21]
            bias_AAs_np (np.array): Global amino acid bias [21]
            chain_M_pos (torch.Tensor): Position-specific design mask [B, L]
            omit_AA_mask (torch.Tensor): Position-specific AA omission [B, L, 21]
            pssm_coef (torch.Tensor): PSSM mixing coefficient [B, L]
            pssm_bias (torch.Tensor): PSSM bias matrix [B, L, 21]
            pssm_multi (float): PSSM weight (0=ignore, 1=only PSSM)
            pssm_log_odds_flag (bool): Use PSSM log-odds filtering
            pssm_log_odds_mask (torch.Tensor): PSSM log-odds mask [B, L, 21]
            pssm_bias_flag (bool): Use PSSM bias
            bias_by_res (torch.Tensor): Per-residue AA bias [B, L, 21]

        Returns:
            dict: Dictionary containing:
                - 'S': Sampled sequences [B, L]
                - 'probs': Sampling probabilities [B, L, 21]
                - 'decoding_order': Order positions were sampled [B, L]

        Sampling Process:
            1. Encode structure with encoder
            2. For each position in random order:
                a. Update decoder hidden states
                b. Compute amino acid logits
                c. Apply temperature, biases, and constraints
                d. Sample amino acid from distribution
                e. Update sequence embedding
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask  #update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21),
                                device=device,
                                dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        #chain_mask_combined = chain_mask*chain_M_pos
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # Sampling step
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]  #[B]
            chain_mask_gathered = torch.gather(chain_mask, 1, t[:, None])  #[B]
            mask_gathered = torch.gather(mask, 1, t[:, None])  #[B]
            bias_by_res_gathered = torch.gather(
                bias_by_res, 1, t[:, None, None].repeat(1, 1,
                                                        21))[:, 0, :]  #[B, 21]
            if (mask_gathered == 0).all():  #for padded or missing regions only
                S_t = torch.gather(S_true, 1, t[:, None])
            else:
                # Hidden layers
                E_idx_t = torch.gather(
                    E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
                h_E_t = torch.gather(
                    h_E, 1, t[:, None, None,
                              None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
                h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                h_EXV_encoder_t = torch.gather(
                    h_EXV_encoder_fw, 1,
                    t[:, None, None,
                      None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                   h_EXV_encoder_fw.shape[-1]))
                mask_t = torch.gather(mask, 1, t[:, None])
                for l, layer in enumerate(self.decoder_layers):
                    # Updated relational features for future states
                    h_ESV_decoder_t = cat_neighbors_nodes(
                        h_V_stack[l], h_ES_t, E_idx_t)
                    h_V_t = torch.gather(
                        h_V_stack[l], 1,
                        t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                    h_ESV_t = torch.gather(
                        mask_bw, 1, t[:, None, None, None].repeat(
                            1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                    ) * h_ESV_decoder_t + h_EXV_encoder_t
                    h_V_stack[l + 1].scatter_(
                        1, t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                        layer(h_V_t, h_ESV_t, mask_V=mask_t))
                # Sampling step
                h_V_t = torch.gather(
                    h_V_stack[-1], 1,
                    t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:,
                                                                            0]
                logits = self.W_out(h_V_t) / temperature
                probs = F.softmax(logits - constant[None, :] * 1e8 +
                                  constant_bias[None, :] / temperature +
                                  bias_by_res_gathered / temperature,
                                  dim=-1)
                
                # PSSM bias
                if pssm_bias_flag:
                    pssm_coef_gathered = torch.gather(pssm_coef, 1,
                                                      t[:, None])[:, 0]
                    pssm_bias_gathered = torch.gather(
                        pssm_bias, 1,
                        t[:, None, None].repeat(1, 1, pssm_bias.shape[-1]))[:,
                                                                            0]
                    probs = (
                        1 - pssm_multi * pssm_coef_gathered[:, None]
                    ) * probs + pssm_multi * pssm_coef_gathered[:,
                                                                None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = torch.gather(
                        pssm_log_odds_mask, 1, t[:, None, None].repeat(
                            1, 1, pssm_log_odds_mask.shape[-1]))[:,
                                                                 0]  #[B, 21]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True)  #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = torch.gather(
                        omit_AA_mask, 1,
                        t[:, None,
                          None].repeat(1, 1,
                                       omit_AA_mask.shape[-1]))[:, 0]  #[B, 21]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True)  #[B, 21]
                S_t = torch.multinomial(probs, 1)
                all_probs.scatter_(1, t[:, None, None].repeat(1, 1, 21),
                                   (chain_mask_gathered[
                                       :,
                                       :,
                                       None,
                                   ] * probs[:, None, :]).float())
            S_true_gathered = torch.gather(S_true, 1, t[:, None])
            S_t = (S_t * chain_mask_gathered + S_true_gathered *
                   (1.0 - chain_mask_gathered)).long()
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]),
                         temp1)
            S.scatter_(1, t[:, None], S_t)
        output_dict = {
            "S": S,
            "probs": all_probs,
            "decoding_order": decoding_order
        }
        return output_dict

    def tied_sample(self,
                    X,
                    randn,
                    S_true,
                    chain_mask,
                    chain_encoding_all,
                    residue_idx,
                    mask=None,
                    temperature=1.0,
                    omit_AAs_np=None,
                    bias_AAs_np=None,
                    chain_M_pos=None,
                    omit_AA_mask=None,
                    pssm_coef=None,
                    pssm_bias=None,
                    pssm_multi=None,
                    pssm_log_odds_flag=None,
                    pssm_log_odds_mask=None,
                    pssm_bias_flag=None,
                    tied_pos=None,
                    tied_beta=None,
                    bias_by_res=None):
        """
        Sample sequences with tied position constraints.

        Similar to sample() but enforces that certain positions must have
        the same amino acid. Critical for designing symmetric proteins,
        homooligomers, or enforcing structural symmetry constraints.

        Args:
            (same as sample() method, plus:)
            tied_pos (list): List of lists of position indices that must be tied
                Example: [[0,10,20], [5,15]] means positions 0,10,20 get same AA
                and positions 5,15 get same AA
            tied_beta (torch.Tensor): Weighting for tied positions [L]
                Controls how strongly each position contributes to tied sampling

        Returns:
            dict: Same as sample() method

        Tied Sampling Process:
            1. Encode structure
            2. Reorder decoding to respect tied constraints
               (all tied positions sampled together)
            3. For each group of tied positions:
                a. Compute logits at all tied positions
                b. Combine logits with tied_beta weights
                c. Sample single amino acid
                d. Assign to all tied positions
            4. Continue with remaining positions

        Use Cases:
            - Symmetric protein design (C2, C3, etc.)
            - Homooligomer design
            - Enforcing sequence repeats
        """
        device = X.device

        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Decoder uses masked self-attention
        chain_mask = chain_mask * chain_M_pos * mask  #update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_mask + 0.0001) * (torch.abs(randn))
        )  #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]

        new_decoding_order = []
        for t_dec in list(decoding_order[
                0,
        ].cpu().data.numpy()):
            if t_dec not in list(itertools.chain(*new_decoding_order)):
                list_a = [item for item in tied_pos if t_dec in item]
                if list_a:
                    new_decoding_order.append(list_a[0])
                else:
                    new_decoding_order.append([t_dec])
        decoding_order = torch.tensor(list(
            itertools.chain(*new_decoding_order)),
                                      device=device)[
                                          None,
                                      ].repeat(X.shape[0], 1)

        # Generate mask
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
            permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = X.size(0), X.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 21), device=device)
        all_probs = torch.zeros((N_batch, N_nodes, 21),
                                device=device,
                                dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=device)
        h_V_stack = [h_V] + [
            torch.zeros_like(h_V, device=device)
            for _ in range(len(self.decoder_layers))
        ]
        constant = torch.tensor(omit_AAs_np, device=device)
        constant_bias = torch.tensor(bias_AAs_np, device=device)
        omit_AA_mask_flag = omit_AA_mask != None

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for t_list in new_decoding_order:
            logits = 0.0
            logit_list = []
            done_flag = False
            for t in t_list:
                if (mask[:, t] == 0).all():
                    S_t = S_true[:, t]
                    for t in t_list:
                        h_S[:, t, :] = self.W_s(S_t)
                        S[:, t] = S_t
                    done_flag = True
                    break
                else:
                    E_idx_t = E_idx[:, t:t + 1, :]
                    h_E_t = h_E[:, t:t + 1, :, :]
                    h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
                    h_EXV_encoder_t = h_EXV_encoder_fw[:, t:t + 1, :, :]
                    mask_t = mask[:, t:t + 1]
                    for l, layer in enumerate(self.decoder_layers):
                        h_ESV_decoder_t = cat_neighbors_nodes(
                            h_V_stack[l], h_ES_t, E_idx_t)
                        h_V_t = h_V_stack[l][:, t:t + 1, :]
                        h_ESV_t = mask_bw[:, t:t +
                                          1, :, :] * h_ESV_decoder_t + h_EXV_encoder_t
                        h_V_stack[l + 1][:, t, :] = layer(
                            h_V_t, h_ESV_t, mask_V=mask_t).squeeze(1)
                    h_V_t = h_V_stack[-1][:, t, :]
                    logit_list.append(
                        (self.W_out(h_V_t) / temperature) / len(t_list))
                    logits += tied_beta[t] * (self.W_out(h_V_t) /
                                              temperature) / len(t_list)
            if done_flag:
                pass
            else:
                bias_by_res_gathered = bias_by_res[:, t, :]  #[B, 21]
                probs = F.softmax(logits - constant[None, :] * 1e8 +
                                  constant_bias[None, :] / temperature +
                                  bias_by_res_gathered / temperature,
                                  dim=-1)
                if pssm_bias_flag:
                    pssm_coef_gathered = pssm_coef[:, t]
                    pssm_bias_gathered = pssm_bias[:, t]
                    probs = (
                        1 - pssm_multi * pssm_coef_gathered[:, None]
                    ) * probs + pssm_multi * pssm_coef_gathered[:,
                                                                None] * pssm_bias_gathered
                if pssm_log_odds_flag:
                    pssm_log_odds_mask_gathered = pssm_log_odds_mask[:, t]
                    probs_masked = probs * pssm_log_odds_mask_gathered
                    probs_masked += probs * 0.001
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True)  #[B, 21]
                if omit_AA_mask_flag:
                    omit_AA_mask_gathered = omit_AA_mask[:, t]
                    probs_masked = probs * (1.0 - omit_AA_mask_gathered)
                    probs = probs_masked / torch.sum(
                        probs_masked, dim=-1, keepdim=True)  #[B, 21]
                S_t_repeat = torch.multinomial(probs, 1).squeeze(-1)
                S_t_repeat = (chain_mask[:, t] * S_t_repeat +
                              (1 - chain_mask[:, t]) *
                              S_true[:, t]).long()  #hard pick fixed positions
                for t in t_list:
                    h_S[:, t, :] = self.W_s(S_t_repeat)
                    S[:, t] = S_t_repeat
                    all_probs[:, t, :] = probs.float()
        output_dict = {
            "S": S,
            "probs": all_probs,
            "decoding_order": decoding_order
        }
        return output_dict

    def conditional_probs(self,
                          X,
                          S,
                          mask,
                          chain_M,
                          residue_idx,
                          chain_encoding_all,
                          randn,
                          backbone_only=False):
        """
        Calculate conditional amino acid probabilities at each position.

        Computes p(AA_i | rest of sequence, backbone) for each position i.
        Useful for analyzing sequence-structure compatibility and predicting
        effects of mutations.

        Args:
            X (torch.Tensor): Backbone coordinates [B, L, 4, 3] or [B, L, 3]
            S (torch.Tensor): Current amino acid sequence [B, L]
            mask (torch.Tensor): Valid residue mask [B, L]
            chain_M (torch.Tensor): Design mask [B, L]
            residue_idx (torch.Tensor): Residue indices [B, L]
            chain_encoding_all (torch.Tensor): Chain IDs [B, L]
            randn (torch.Tensor): Random noise for ordering [B, L]
            backbone_only (bool): If True, condition only on backbone (no sequence)
                If False, condition on both backbone and rest of sequence

        Returns:
            torch.Tensor: Log conditional probabilities [B, L, 21]

        Process:
            For each position i:
            1. Mask position i as "unknown"
            2. Run decoder with all other positions known
            3. Record p(AA_i | everything else)

        Applications:
            - Mutation effect prediction
            - Sequence-structure compatibility analysis
            - Identifying challenging positions
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V_enc = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]),
                              device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V_enc, h_E = layer(h_V_enc, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  #update chain_M to include missing regions

        chain_M_np = chain_M.cpu().numpy()
        idx_to_loop = np.argwhere(chain_M_np[0, :] == 1)[:, 0]
        log_conditional_probs = torch.zeros([X.shape[0], chain_M.shape[1], 21],
                                            device=device).float()

        for idx in idx_to_loop:
            h_V = torch.clone(h_V_enc)
            order_mask = torch.zeros(chain_M.shape[1], device=device).float()
            if backbone_only:
                order_mask = torch.ones(chain_M.shape[1],
                                        device=device).float()
                order_mask[idx] = 0.
            else:
                order_mask = torch.zeros(chain_M.shape[1],
                                         device=device).float()
                order_mask[idx] = 1.
            decoding_order = torch.argsort(
                (order_mask[
                    None,
                ] + 0.0001) * (torch.abs(randn))
            )  #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
            mask_size = E_idx.shape[1]
            permutation_matrix_reverse = torch.nn.functional.one_hot(
                decoding_order, num_classes=mask_size).float()
            order_mask_backward = torch.einsum(
                'ij, biq, bjp->bqp',
                (1 -
                 torch.triu(torch.ones(mask_size, mask_size, device=device))),
                permutation_matrix_reverse, permutation_matrix_reverse)
            mask_attend = torch.gather(order_mask_backward, 2,
                                       E_idx).unsqueeze(-1)
            mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
            mask_bw = mask_1D * mask_attend
            mask_fw = mask_1D * (1. - mask_attend)

            h_EXV_encoder_fw = mask_fw * h_EXV_encoder
            for layer in self.decoder_layers:
                # Masked positions attend to encoder information, unmasked see.
                h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
                h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
                h_V = layer(h_V, h_ESV, mask)

            logits = self.W_out(h_V)
            log_probs = F.log_softmax(logits, dim=-1)
            log_conditional_probs[:, idx, :] = log_probs[:, idx, :]
        return log_conditional_probs

    def unconditional_probs(self, X, mask, residue_idx, chain_encoding_all):
        """
        Calculate unconditional amino acid probabilities from backbone only.

        Computes p(AA_i | backbone) with no sequence information.
        All positions are predicted independently based purely on structure.

        Args:
            X (torch.Tensor): Backbone coordinates [B, L, 4, 3] or [B, L, 3]
            mask (torch.Tensor): Valid residue mask [B, L]
            residue_idx (torch.Tensor): Residue indices [B, L]
            chain_encoding_all (torch.Tensor): Chain IDs [B, L]

        Returns:
            torch.Tensor: Log unconditional probabilities [B, L, 21]

        Process:
            1. Encode backbone structure
            2. Decode with no sequence information (all positions masked)
            3. Predict amino acid probabilities independently at each position

        Differences from conditional_probs:
            - No sequence context used
            - All positions predicted in parallel (not autoregressively)
            - Faster but less accurate than conditional probabilities

        Applications:
            - Quick structure-based sequence prediction
            - Baseline for sequence design
            - Identifying structurally constrained positions
        """
        device = X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]),
                          device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        order_mask_backward = torch.zeros([X.shape[0], X.shape[1], X.shape[1]],
                                          device=device)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_EXV_encoder_fw, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
