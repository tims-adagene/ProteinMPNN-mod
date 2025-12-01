"""
Utility functions and classes for protein structure data loading and processing.

This module provides data loading infrastructure for ProteinMPNN training, including:
- Dataset classes for handling protein structure data
- Data loaders with length-based batching for efficiency
- PDB file loading and structure assembly functions
- Training/validation/test split management
- Worker initialization for parallel data loading
"""

import torch
from torch.utils.data import DataLoader
import csv
from dateutil import parser
import numpy as np
import time
import random
import os


class StructureDataset():
    """
    PyTorch Dataset for protein structures with validation and filtering.

    Filters protein structures by sequence validity and length, providing
    iteration over protein complexes for training.
    """

    def __init__(self,
                 pdb_dict_list,
                 verbose=True,
                 truncate=None,
                 max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        """
        Initialize structure dataset.

        Parameters
        ----------
        pdb_dict_list : list of dict
            List of protein structure dictionaries with keys:
            - 'seq': amino acid sequence
            - 'name': structure identifier
            - 'xyz': atomic coordinates
            - 'idx': chain indices
            - etc.
        verbose : bool
            Print progress information
        truncate : int, optional
            Maximum number of structures to load (for debugging)
        max_length : int
            Maximum sequence length to include
        alphabet : str
            Allowed amino acid characters
        """
        alphabet_set = set([a for a in alphabet])
        discard_count = {'bad_chars': 0, 'too_long': 0, 'bad_seq_length': 0}

        self.data = []

        start = time.time()
        # Filter and validate structures
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            # Check for invalid characters in sequence
            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                # Accept sequences within length limit
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                # Skip sequences with invalid characters
                discard_count['bad_chars'] += 1

            # Truncate early if specified (for debugging)
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                # Progress update every 1000 structures

    def __len__(self):
        """Return number of structures in dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get structure at given index."""
        return self.data[idx]


class StructureLoader():
    """
    Data loader that batches protein structures by similar lengths.

    Creates batches of structures with similar sequence lengths to minimize
    padding and improve computational efficiency.
    """

    def __init__(self,
                 dataset,
                 batch_size=100,
                 shuffle=True,
                 collate_fn=lambda x: x,
                 drop_last=False):
        """
        Initialize structure loader with length-based batching.

        Parameters
        ----------
        dataset : StructureDataset
            Dataset of protein structures
        batch_size : int
            Maximum batch size in tokens (sum of sequence lengths)
        shuffle : bool
            Whether to shuffle batch order (always on for epochs)
        collate_fn : callable, optional
            Custom collate function (unused, keeps default)
        drop_last : bool, optional
            Whether to drop last incomplete batch (unused)
        """
        self.dataset = dataset
        self.size = len(dataset)
        # Extract sequence lengths for all structures
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        # Sort indices by sequence length for batching
        sorted_ix = np.argsort(self.lengths)

        # Cluster structures into batches of similar sizes
        # Batches are created by adding structures until token limit is reached
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            # Add to batch if within batch_size token limit
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                # Start new batch if limit exceeded
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        """Return number of batches."""
        return len(self.clusters)

    def __iter__(self):
        """Iterate over batches, shuffling their order each epoch."""
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            # Retrieve structures for this batch
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def worker_init_fn(worker_id):
    """
    Initialize random seed for data loading worker.

    Called once per worker process to ensure different random sequences
    across parallel data loaders.

    Parameters
    ----------
    worker_id : int
        Worker process ID (set automatically by DataLoader)
    """
    np.random.seed()


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)


def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    """
    Load protein structures from PDB data loader and extract chain information.

    Loads PDB structures from a DataLoader, extracts individual chains, removes
    His-tags, and concatenates multi-chain complexes for training.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader providing PDB entries with coordinates and sequences
    repeat : int
        Number of times to iterate through data_loader (for resampling)
    max_length : int
        Maximum protein length to include
    num_units : int
        Maximum number of individual chain instances to load

    Returns
    -------
    list of dict
        List of protein structures with keys:
        - 'seq': concatenated sequence
        - 'xyz': atomic coordinates
        - 'idx': chain indices
        - 'masked': homologous chain indices
        - 'label': structure identifier
    """
    # Create alphabet for chain naming (can handle up to 352 chains)
    init_alphabet = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
        'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c = 0
    c1 = 0
    pdb_dict_list = []
    t0 = time.time()
    for _ in range(repeat):
        for step, t in enumerate(data_loader):
            # Extract single element from batch dimension
            t = {k: v[0] for k, v in t.items()}
            c1 += 1
            # Process structures that have a label field
            if 'label' in list(t):
                my_dict = {}
                s = 0
                concat_seq = ''
                concat_N = []
                concat_CA = []
                concat_C = []
                concat_O = []
                concat_mask = []
                coords_dict = {}
                mask_list = []
                visible_list = []
                # Process multi-chain complexes (up to 352 chains)
                if len(list(np.unique(t['idx']))) < 352:
                    # Iterate over unique chains in the structure
                    for idx in list(np.unique(t['idx'])):
                        letter = chain_alphabet[idx]
                        # Get residue indices for this chain
                        res = np.argwhere(t['idx'] == idx)
                        # Extract sequence for this chain
                        initial_sequence = "".join(
                            list(np.array(list(t['seq']))[res][
                                0,
                            ]))
                        # Remove His-tags (6 consecutive H residues)
                        # Check and remove C-terminal His-tag
                        if initial_sequence[-6:] == "HHHHHH":
                            res = res[:, :-6]
                        # Check and remove N-terminal His-tag
                        if initial_sequence[0:6] == "HHHHHH":
                            res = res[:, 6:]
                        if initial_sequence[-7:-1] == "HHHHHH":
                            res = res[:, :-7]
                        if initial_sequence[-8:-2] == "HHHHHH":
                            res = res[:, :-8]
                        if initial_sequence[-9:-3] == "HHHHHH":
                            res = res[:, :-9]
                        if initial_sequence[-10:-4] == "HHHHHH":
                            res = res[:, :-10]
                        if initial_sequence[1:7] == "HHHHHH":
                            res = res[:, 7:]
                        if initial_sequence[2:8] == "HHHHHH":
                            res = res[:, 8:]
                        if initial_sequence[3:9] == "HHHHHH":
                            res = res[:, 9:]
                        if initial_sequence[4:10] == "HHHHHH":
                            res = res[:, 10:]
                        if res.shape[1] < 4:
                            pass
                        else:
                            my_dict['seq_chain_' + letter] = "".join(
                                list(np.array(list(t['seq']))[res][
                                    0,
                                ]))
                            concat_seq += my_dict['seq_chain_' + letter]
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][
                                res,
                            ])[
                                0,
                            ]  #[L, 14, 3]
                            coords_dict_chain[
                                'N_chain_' +
                                letter] = all_atoms[:, 0, :].tolist()
                            coords_dict_chain[
                                'CA_chain_' +
                                letter] = all_atoms[:, 1, :].tolist()
                            coords_dict_chain[
                                'C_chain_' +
                                letter] = all_atoms[:, 2, :].tolist()
                            coords_dict_chain[
                                'O_chain_' +
                                letter] = all_atoms[:, 3, :].tolist()
                            my_dict['coords_chain_' +
                                    letter] = coords_dict_chain
                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(
                        visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list


class PDB_dataset(torch.utils.data.Dataset):

    def __init__(self, IDs, loader, train_dict, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        return out


def loader_pdb(item, params):
    """
    Load a single PDB structure with its assembly information.

    Loads protein coordinates and sequence for a specific chain, optionally
    including biological assembly information with homologous chain identification.

    Parameters
    ----------
    item : tuple
        (PDB_ID_CHAIN_ID, additional_info) where PDB_ID_CHAIN_ID format is "PDBID_CHAINID"
    params : dict
        Configuration parameters with keys:
        - 'DIR': base data directory
        - 'HOMO': sequence identity threshold for homolog detection

    Returns
    -------
    dict
        Structure data with keys:
        - 'seq': amino acid sequence
        - 'xyz': atomic coordinates [L, 14, 3]
        - 'idx': chain indices
        - 'masked': homologous chain indices
        - 'label': structure identifier
    """
    # Parse PDB ID and chain ID from item
    pdbid, chid = item[0].split('_')
    # Construct file path: pdb_dir/pdb/XX/PDBID where XX is 2nd-3rd chars of PDBID
    PREFIX = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)

    # Load metadata and coordinate files
    if not os.path.isfile(PREFIX + ".pt"):
        return {'seq': np.zeros(5)}  # Return dummy structure if file not found
    meta = torch.load(PREFIX + ".pt")  # Load metadata (assembly info, etc.)
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = set(
        [a for a, b in zip(asmb_ids, asmb_chains) if chid in b.split(',')])

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if len(asmb_candidates) < 1:
        chain = torch.load("%s_%s.pt" % (PREFIX, chid))
        L = len(chain['seq'])
        return {
            'seq': chain['seq'],
            'xyz': chain['xyz'],
            'idx': torch.zeros(L).int(),
            'masked': torch.Tensor([0]).int(),
            'label': item[0]
        }

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    # load relevant chains
    chains = {
        c: torch.load("%s_%s.pt" % (PREFIX, c))
        for i in idx
        for c in asmb_chains[i] if c in meta['chains']
    }

    # generate assembly
    asmb = {}
    for k in idx:

        # pick k-th xform
        xform = meta['asmb_xform%d' % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        # transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:, None,
                                                                   None, :]
                asmb.update({
                    (c, k, i): xyz_i
                    for i, xyz_i in enumerate(xyz_ru)
                })
            except KeyError:
                return {'seq': np.zeros(5)}

    # select chains which share considerable similarity to chid
    seqid = meta['tm'][chids == chid][0, :, 1]
    homo = set([
        ch_j for seqid_j, ch_j in zip(seqid, chids) if seqid_j > params['HOMO']
    ])
    # stack all chains in the assembly together
    seq, xyz, idx, masked = "", [], [], []
    seq_list = []
    for counter, (k, v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        seq_list.append(chains[k[0]]['seq'])
        xyz.append(v)
        idx.append(torch.full((v.shape[0], ), counter))
        if k[0] in homo:
            masked.append(counter)

    return {
        'seq': seq,
        'xyz': torch.cat(xyz, dim=0),
        'idx': torch.cat(idx, dim=0),
        'masked': torch.Tensor(masked).int(),
        'label': item[0]
    }


def build_training_clusters(params, debug):
    """
    Build training, validation, and test datasets from PDB list file.

    Reads a CSV file with PDB metadata, filters by resolution and date,
    and splits into training, validation, and test sets using sequence
    similarity clustering.

    Parameters
    ----------
    params : dict
        Configuration parameters with keys:
        - 'LIST': path to CSV file with PDB metadata
        - 'VAL': path to file with validation cluster IDs
        - 'TEST': path to file with test cluster IDs
        - 'RESCUT': maximum resolution cutoff (Angstroms)
        - 'DATCUT': date cutoff (ISO format)
    debug : bool
        If True, use small subset for debugging

    Returns
    -------
    tuple
        - train: dict mapping cluster_id to list of [PDB_ID, CHAIN_ID] pairs
        - valid: dict mapping cluster_id to list of [PDB_ID, CHAIN_ID] pairs
        - test: dict mapping cluster_id to list of [PDB_ID, CHAIN_ID] pairs
    """
    # Load validation and test cluster IDs
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])

    # Override with empty sets for debugging (use all as training)
    if debug:
        val_ids = []
        test_ids = []

    # Read and filter PDB metadata from CSV file
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        # Filter by resolution and deposition date
        rows = [[r[0], r[3], int(r[4])] for r in reader
                if float(r[2]) <= params['RESCUT']  # Resolution cutoff
                and parser.parse(r[1]) <= parser.parse(params['DATCUT'])]  # Date cutoff

    # Initialize dataset dictionaries
    train = {}
    valid = {}
    test = {}

    # Use small subset for debugging
    if debug:
        rows = rows[:20]

    # Distribute PDB entries to train/val/test based on cluster IDs
    for r in rows:
        cluster_id = r[2]  # Cluster ID for sequence similarity clustering
        pdb_chain = r[:2]  # [PDB_ID, CHAIN_ID]
        # Assign to appropriate split
        if cluster_id in val_ids:
            if cluster_id in valid.keys():
                valid[cluster_id].append(pdb_chain)
            else:
                valid[cluster_id] = [pdb_chain]
        elif cluster_id in test_ids:
            if cluster_id in test.keys():
                test[cluster_id].append(pdb_chain)
            else:
                test[cluster_id] = [pdb_chain]
        else:
            if cluster_id in train.keys():
                train[cluster_id].append(pdb_chain)
            else:
                train[cluster_id] = [pdb_chain]

    # Use training set as validation for debugging
    if debug:
        valid = train
    return train, valid, test
