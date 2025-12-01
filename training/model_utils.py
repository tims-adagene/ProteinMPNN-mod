"""
Model utility functions for ProteinMPNN training.

This module contains core functions and classes for the ProteinMPNN model, including:
- Feature extraction and featurization of protein batches
- Loss computation functions (NLL and label-smoothed losses)
- Graph gathering functions for neighbor aggregation
- Transformer-based encoder and decoder layers
- Protein feature extraction and RBF computations
- The main ProteinMPNN model architecture
- Noam optimizer implementation for learning rate scheduling

The module supports training on GPU with mixed precision and gradient checkpointing
for memory efficiency.
"""

from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


def featurize(batch, device):
    """
    Convert a batch of protein structures into tensor features for model input.

    This function processes a batch of protein structure dictionaries and extracts:
    - Backbone coordinates (N, CA, C, O atoms)
    - Sequence information encoded as integers
    - Mask information for which residues need to be predicted
    - Chain encoding to distinguish different chains
    - Residue indexing with chain offsets
    - Self-interaction mask for chain boundaries

    Parameters
    ----------
    batch : list of dict
        List of protein structure dictionaries, each containing:
        - 'seq': concatenated sequence string
        - 'masked_list': list of chain IDs to be masked (predicted)
        - 'visible_list': list of chain IDs to be visible (observed)
        - 'seq_chain_X': sequence for chain X
        - 'coords_chain_X': dict with backbone atom coordinates for chain X
        - 'num_of_chains': number of chains
    device : torch.device
        Device to place tensors on (CPU or CUDA)

    Returns
    -------
    tuple
        - X: backbone coordinates [B, L, 4, 3] (batch, length, 4 atoms, xyz)
        - S: amino acid sequence labels [B, L]
        - mask: valid positions [B, L] (1.0 for atoms with coordinates)
        - lengths: sequence lengths per sample [B]
        - chain_M: masking indicator [B, L] (1.0 for positions to predict)
        - residue_idx: residue indexing with chain offsets [B, L]
        - mask_self: chain boundary mask [B, L, L]
        - chain_encoding_all: chain assignment [B, L]
    """
    # Define amino acid alphabet and initialize batch dimensions
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)

    # Calculate sequence lengths for each sample in batch
    lengths = np.array([len(b['seq']) for b in batch],
                       dtype=np.int32)  # sum of all chain seq lengths
    L_max = max([len(b['seq']) for b in batch])  # Maximum sequence length

    # Initialize output tensors with batch and maximum sequence length
    X = np.zeros([B, L_max, 4, 3])  # Backbone atom coordinates (N, CA, C, O)
    residue_idx = -100 * np.ones(
        [B, L_max], dtype=np.int32)  # Residue indexing with chain offsets
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # Masking indicator: 1.0 for positions to predict, 0.0 for given
    mask_self = np.ones(
        [B, L_max, L_max], dtype=np.int32
    )  # Chain boundary mask: 0.0 for intra-chain, 1.0 for inter-chain
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # Chain ID assignment (0 for chain A, 1 for chain B, etc.)
    S = np.zeros([B, L_max], dtype=np.int32)  # Amino acid sequence labels
    # Create extended alphabet to support up to 352 chains
    init_alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet

    # Process each sample in the batch
    for i, b in enumerate(batch):
        # Extract masked (to be predicted) and visible (observed) chains
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}

        # Build sequence dictionaries for quick lookup
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq

        # Resolve duplicate sequences: if a masked chain has same sequence as visible chain,
        # mark the visible chain as masked and vice versa
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)

        # Update chain lists and randomly shuffle order for diversity
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # Randomly shuffle chain order for training variation
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[
                    f'coords_chain_{letter}']  #this is a dictionary
                chain_mask = np.zeros(chain_length)  #0.0 for visible chains
                x_chain = np.stack([
                    chain_coords[c] for c in [
                        f'N_chain_{letter}', f'CA_chain_{letter}',
                        f'C_chain_{letter}', f'O_chain_{letter}'
                    ]
                ], 1)  #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(
                    c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1,
                          l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[
                    f'coords_chain_{letter}']  #this is a dictionary
                chain_mask = np.ones(chain_length)  #0.0 for visible chains
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
                mask_self[i, l0:l1,
                          l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
        x = np.concatenate(x_chain_list, 0)  #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,
                           0)  #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]],
                       'constant',
                       constant_values=(np.nan, ))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]],
                       'constant',
                       constant_values=(0.0, ))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]],
                                    'constant',
                                    constant_values=(0.0, ))
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence],
                             dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,
                                                   device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32,
                                               device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(
        dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


def loss_nll(S, log_probs, mask):
    """
    Compute negative log likelihood loss with accuracy metrics.

    Parameters
    ----------
    S : torch.Tensor
        Ground truth amino acid labels [B, L]
    log_probs : torch.Tensor
        Log probabilities from model [B, L, 21]
    mask : torch.Tensor
        Valid position mask [B, L]

    Returns
    -------
    tuple
        - loss: per-position loss [B, L]
        - loss_av: masked average loss (scalar)
        - true_false: accuracy indicator [B, L] (1.0 for correct predictions)
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    # Reshape for loss computation and reshape back to original dimensions
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)),
                     S.contiguous().view(-1)).view(S.size())
    # Compute predictions and accuracy
    S_argmaxed = torch.argmax(log_probs, -1)  # Get most likely amino acid [B, L]
    true_false = (S == S_argmaxed).float()  # 1.0 where prediction is correct
    # Compute masked average loss
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """
    Compute label-smoothed cross-entropy loss.

    Label smoothing improves model generalization by preventing overconfident predictions.
    The ground truth distribution is smoothed by adding uniform probability mass.

    Parameters
    ----------
    S : torch.Tensor
        Ground truth amino acid labels [B, L]
    log_probs : torch.Tensor
        Log probabilities from model [B, L, 21]
    mask : torch.Tensor
        Valid position mask [B, L]
    weight : float
        Label smoothing weight (default 0.1)

    Returns
    -------
    tuple
        - loss: per-position loss [B, L]
        - loss_av: masked average loss (scalar)
    """
    # Convert labels to one-hot encoding
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Apply label smoothing: add uniform probability to all classes
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    # Renormalize to sum to 1
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    # Compute cross-entropy loss
    loss = -(S_onehot * log_probs).sum(-1)
    # Compute masked average loss (fixed denominator of 2000 for stability)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av


# Graph neighbor aggregation functions
def gather_edges(edges, neighbor_idx):
    """
    Gather edge features for neighbor indices.

    Parameters
    ----------
    edges : torch.Tensor
        Edge features [B, N, N, C] (batch, nodes, nodes, channels)
    neighbor_idx : torch.Tensor
        Neighbor indices [B, N, K]

    Returns
    -------
    torch.Tensor
        Gathered edge features [B, N, K, C] for neighbors of each node
    """
    # Expand indices to match feature dimension
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    # Gather features at neighbor indices
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    """
    Gather node features for neighbor indices.

    Parameters
    ----------
    nodes : torch.Tensor
        Node features [B, N, C] (batch, nodes, channels)
    neighbor_idx : torch.Tensor
        Neighbor indices [B, N, K]

    Returns
    -------
    torch.Tensor
        Gathered node features [B, N, K, C] for neighbors of each node
    """
    # Flatten neighbor indices to batch dimension
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    # Expand indices to match feature dimension
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather node features at flattened neighbor indices
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    # Reshape back to [B, N, K, C]
    neighbor_features = neighbor_features.view(
        list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    """
    Gather node features for single neighbor index per batch.

    Parameters
    ----------
    nodes : torch.Tensor
        Node features [B, N, C]
    neighbor_idx : torch.Tensor
        Neighbor indices [B, K]

    Returns
    -------
    torch.Tensor
        Gathered node features [B, K, C]
    """
    # Expand indices to match feature dimension
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather node features
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    """
    Concatenate node features with their neighbor features.

    Parameters
    ----------
    h_nodes : torch.Tensor
        Node features [B, N, C]
    h_neighbors : torch.Tensor
        Neighbor features [B, N, K, C]
    E_idx : torch.Tensor
        Neighbor indices [B, N, K]

    Returns
    -------
    torch.Tensor
        Concatenated features [B, N, K, 2*C]
    """
    # Gather self-node features repeated for each neighbor
    h_nodes = gather_nodes(h_nodes, E_idx)
    # Concatenate neighbor and node features
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    """
    Graph Transformer encoder layer for protein structure processing.

    This layer performs message passing on a graph-structured representation of protein
    structures, updating both node (residue) and edge (pair) features through multiple
    sub-layers with layer normalization and dropout.
    """

    def __init__(self,
                 num_hidden,
                 num_in,
                 dropout=0.1,
                 num_heads=None,
                 scale=30):
        """
        Initialize encoder layer.

        Parameters
        ----------
        num_hidden : int
            Dimension of node features
        num_in : int
            Dimension of edge features
        dropout : float
            Dropout probability
        num_heads : int, optional
            Unused parameter (kept for compatibility)
        scale : float
            Scaling factor for aggregation (default 30)
        """
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        # Node update network (processes concatenated node and edge features)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        # Edge update network
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        # Activation function
        self.act = torch.nn.GELU()
        # Position-wise feedforward network
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """
        Forward pass for encoder layer.

        Parameters
        ----------
        h_V : torch.Tensor
            Node features [B, N, C]
        h_E : torch.Tensor
            Edge features [B, N, K, C]
        E_idx : torch.Tensor
            Neighbor indices [B, N, K]
        mask_V : torch.Tensor, optional
            Node mask [B, N]
        mask_attend : torch.Tensor, optional
            Attention mask [B, N, K]

        Returns
        -------
        tuple
            - h_V: updated node features [B, N, C]
            - h_E: updated edge features [B, N, K, C]
        """
        # First sublayer: aggregate neighbor messages to update node features
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        # Message computation with MLPs
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        # Apply attention mask if provided
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        # Aggregate messages and update node features
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Second sublayer: position-wise feedforward network
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        # Apply node mask if provided
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        # Third sublayer: update edge features based on updated nodes
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        # Edge message computation
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    """
    Graph Transformer decoder layer for protein sequence generation.

    This layer is used in the autoregressive decoding phase, processing sequence
    embeddings and edge features to predict amino acid identities at each position.
    """

    def __init__(self,
                 num_hidden,
                 num_in,
                 dropout=0.1,
                 num_heads=None,
                 scale=30):
        """
        Initialize decoder layer.

        Parameters
        ----------
        num_hidden : int
            Dimension of node features
        num_in : int
            Dimension of edge features
        dropout : float
            Dropout probability
        num_heads : int, optional
            Unused parameter (kept for compatibility)
        scale : float
            Scaling factor for aggregation (default 30)
        """
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        # Dropout and normalization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        # Message passing networks
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        # Position-wise feedforward network
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """
        Forward pass for decoder layer.

        Parameters
        ----------
        h_V : torch.Tensor
            Node features [B, N, C]
        h_E : torch.Tensor
            Edge features [B, N, K, C]
        mask_V : torch.Tensor, optional
            Node mask [B, N]
        mask_attend : torch.Tensor, optional
            Attention mask [B, N, K]

        Returns
        -------
        torch.Tensor
            Updated node features [B, N, C]
        """
        # Message passing: concatenate current node features with neighbor edge features
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        # Compute messages through MLPs
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        # Apply attention mask if provided
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        # Aggregate messages with scaling
        dh = torch.sum(h_message, -2) / self.scale

        # Update node features with residual connection and normalization
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward sub-layer
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        # Apply node mask if provided
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise feedforward network used in transformer layers.

    Standard FFN with input projection to higher dimension, GELU activation,
    and output projection back to original dimension.
    """

    def __init__(self, num_hidden, num_ff):
        """
        Initialize feedforward network.

        Parameters
        ----------
        num_hidden : int
            Input and output dimension
        num_ff : int
            Hidden dimension (typically 4x num_hidden)
        """
        super(PositionWiseFeedForward, self).__init__()
        # Expand to higher dimension
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        # Project back to original dimension
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        """Apply feedforward transformation."""
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(nn.Module):
    """
    Relative positional encoding for sequence positions.

    Encodes the relative distance between positions in the protein sequence,
    distinguishing intra-chain and inter-chain relationships.
    """

    def __init__(self, num_embeddings, max_relative_feature=32):
        """
        Initialize positional encoding.

        Parameters
        ----------
        num_embeddings : int
            Output embedding dimension
        max_relative_feature : int
            Maximum relative distance to encode (default 32)
        """
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        # Linear layer to project one-hot positions to embeddings
        self.linear = nn.Linear(2 * max_relative_feature + 1 + 1,
                                num_embeddings)

    def forward(self, offset, mask):
        """
        Compute positional embeddings for relative distances.

        Parameters
        ----------
        offset : torch.Tensor
            Relative position offsets [B, N, K]
        mask : torch.Tensor
            Chain boundary mask (1 for same chain, 0 for different) [B, N, K]

        Returns
        -------
        torch.Tensor
            Positional embeddings [B, N, K, C]
        """
        # Clip relative distances to valid range, use special value for different chains
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask + (1 - mask) * (
                           2 * self.max_relative_feature + 1)
        # Convert to one-hot representation
        d_onehot = torch.nn.functional.one_hot(
            d, 2 * self.max_relative_feature + 1 + 1)
        # Project one-hot to embedding space
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    """
    Extract and embed geometric features from protein structures.

    Computes graph edges based on spatial distances, radial basis function (RBF)
    kernels for distance encoding, and relative positional embeddings for
    sequence positions.
    """

    def __init__(self,
                 edge_features,
                 node_features,
                 num_positional_embeddings=16,
                 num_rbf=16,
                 top_k=30,
                 augment_eps=0.,
                 num_chain_embeddings=16):
        """
        Initialize protein feature extractor.

        Parameters
        ----------
        edge_features : int
            Output dimension for edge embeddings
        node_features : int
            Output dimension for node embeddings
        num_positional_embeddings : int
            Dimension of positional embeddings (default 16)
        num_rbf : int
            Number of RBF kernels for distance encoding (default 16)
        top_k : int
            Number of nearest neighbors for graph edges (default 30)
        augment_eps : float
            Coordinate perturbation for data augmentation (default 0.0)
        num_chain_embeddings : int
            Unused parameter (default 16)
        """
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
            torch.sum((A[:, :, None, :] - B[:, None, :, :])**2, -1) +
            1e-6)  #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None],
                                       E_idx)[:, :, :, 0]  #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
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

    def __init__(self,
                 num_letters=21,
                 node_features=128,
                 edge_features=128,
                 hidden_dim=128,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 vocab=21,
                 k_neighbors=32,
                 augment_eps=0.1,
                 dropout=0.1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

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

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
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
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask  #update chain_M to include missing regions
        decoding_order = torch.argsort(
            (chain_M + 0.0001) *
            (torch.abs(torch.randn(chain_M.shape, device=device)))
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
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


class NoamOpt:
    """
    Optimizer wrapper implementing Noam learning rate scheduling.

    The learning rate follows: lr = factor * (d_model^-0.5) * min(step^-0.5, step * warmup^-1.5)
    This schedule increases learning rate during warmup phase then decays it afterwards.
    """

    def __init__(self, model_size, factor, warmup, optimizer, step):
        """
        Initialize Noam optimizer.

        Parameters
        ----------
        model_size : int
            Model hidden dimension (used for scaling)
        factor : float
            Scaling factor for learning rate
        warmup : int
            Number of warmup steps
        optimizer : torch.optim.Optimizer
            Underlying PyTorch optimizer
        step : int
            Starting step number (for resuming training)
        """
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return optimizer param groups for compatibility."""
        return self.optimizer.param_groups

    def step(self):
        """Update model parameters and learning rate."""
        self._step += 1
        # Compute new learning rate based on current step
        rate = self.rate()
        # Update learning rate for all parameter groups
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # Perform optimizer step
        self.optimizer.step()

    def rate(self, step=None):
        """
        Compute learning rate for given training step.

        Parameters
        ----------
        step : int, optional
            Training step (default uses current step)

        Returns
        -------
        float
            Learning rate for the given step
        """
        if step is None:
            step = self._step
        # Noam learning rate schedule
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        """Clear gradients in underlying optimizer."""
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    """
    Create a standard Noam-scheduled optimizer.

    Parameters
    ----------
    parameters : iterable
        Model parameters to optimize
    d_model : int
        Model hidden dimension
    step : int
        Starting training step

    Returns
    -------
    NoamOpt
        Optimizer with Noam learning rate schedule
    """
    return NoamOpt(
        d_model, 2, 4000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)
