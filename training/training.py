"""
Training script for ProteinMPNN model.

This script handles the complete training loop for the ProteinMPNN model, including:
- Data loading and batch preparation
- Model training with mixed precision support
- Validation and evaluation
- Learning rate scheduling
- Checkpoint saving and resuming

The script uses gradient accumulation for large batch sizes and supports
distributed data loading with multiple worker processes.
"""

import argparse
import os.path


def main(args):
    """
    Main training function for ProteinMPNN model.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing training configuration:
        - path_for_training_data: path to PDB dataset
        - path_for_outputs: output directory for logs and checkpoints
        - previous_checkpoint: path to resume from checkpoint
        - num_epochs: number of training epochs
        - batch_size: training batch size
        - max_protein_length: maximum protein length
        - hidden_dim: model hidden dimension
        - num_encoder_layers: number of encoder layers
        - num_decoder_layers: number of decoder layers
        - num_neighbors: number of k-nearest neighbors for graph
        - dropout: dropout probability
        - backbone_noise: coordinate augmentation noise
        - mixed_precision: use mixed precision training
        - gradient_norm: gradient clipping threshold
    """
    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    import queue
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from utils import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset, StructureDataset, StructureLoader
    from model_utils import featurize, loss_smoothed, loss_nll, get_std_opt, ProteinMPNN

    # Initialize mixed precision gradient scaler for automatic scaling
    scaler = torch.cuda.amp.GradScaler()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Create output directory with timestamp
    base_folder = time.strftime(args.path_for_outputs, time.localtime())

    if base_folder[-1] != '/':
        base_folder += '/'
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    # Create subdirectory for model checkpoints
    subfolders = ['model_weights']
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    # Load previous checkpoint if resuming training
    PATH = args.previous_checkpoint

    # Initialize or append to training log
    logfile = base_folder + 'log.txt'
    if not PATH:
        with open(logfile, 'w') as f:
            f.write('Epoch\tTrain\tValidation\n')

    # Load training dataset parameters
    data_path = args.path_for_training_data
    params = {
        "LIST": f"{data_path}/list.csv",  # CSV file with PDB metadata
        "VAL": f"{data_path}/valid_clusters.txt",  # Validation cluster IDs
        "TEST": f"{data_path}/test_clusters.txt",  # Test cluster IDs
        "DIR": f"{data_path}",  # Base data directory
        "DATCUT": "2030-Jan-01",  # Date cutoff (include all structures up to this date)
        "RESCUT": args.rescut,  # Resolution cutoff for PDBs (in Angstroms)
        "HOMO": 0.70  # Sequence identity threshold for homologous chain detection
    }

    # DataLoader parameters
    LOAD_PARAM = {
        'batch_size': 1,  # Load one PDB at a time (will be split into chains)
        'shuffle': True,  # Shuffle order of PDBs
        'pin_memory': False,  # Pin memory for faster GPU transfer
        'num_workers': 4  # Parallel workers for data loading
    }

    # Override parameters for debugging
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    # Build training, validation, and test datasets
    train, valid, test = build_training_clusters(params, args.debug)

    # Create training and validation datasets
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               worker_init_fn=worker_init_fn,
                                               **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               worker_init_fn=worker_init_fn,
                                               **LOAD_PARAM)

    # Initialize ProteinMPNN model with specified hyperparameters
    model = ProteinMPNN(node_features=args.hidden_dim,
                        edge_features=args.hidden_dim,
                        hidden_dim=args.hidden_dim,
                        num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_encoder_layers,
                        k_neighbors=args.num_neighbors,
                        dropout=args.dropout,
                        augment_eps=args.backbone_noise)
    model.to(device)

    # Load checkpoint if resuming training
    if PATH:
        checkpoint = torch.load(PATH)
        total_step = checkpoint['step']  # Restore training step counter
        epoch = checkpoint['epoch']  # Restore epoch counter
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        total_step = 0
        epoch = 0

    # Initialize optimizer with Noam learning rate scheduling
    optimizer = get_std_opt(model.parameters(), args.hidden_dim, total_step)

    if PATH:
        optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Main training loop with parallel data loading
    with ProcessPoolExecutor(max_workers=12) as executor:
        q = queue.Queue(maxsize=3)
        p = queue.Queue(maxsize=3)
        # Pre-load datasets in parallel to avoid blocking on data loading
        for i in range(3):
            q.put_nowait(
                executor.submit(get_pdbs, train_loader, 1,
                                args.max_protein_length,
                                args.num_examples_per_epoch))
            p.put_nowait(
                executor.submit(get_pdbs, valid_loader, 1,
                                args.max_protein_length,
                                args.num_examples_per_epoch))
        # Get initial datasets while background loading continues
        pdb_dict_train = q.get().result()
        pdb_dict_valid = p.get().result()

        # Create datasets with length filtering
        dataset_train = StructureDataset(pdb_dict_train,
                                         truncate=None,
                                         max_length=args.max_protein_length)
        dataset_valid = StructureDataset(pdb_dict_valid,
                                         truncate=None,
                                         max_length=args.max_protein_length)

        # Create loaders that batch by similar protein lengths for efficiency
        loader_train = StructureLoader(dataset_train,
                                       batch_size=args.batch_size)
        loader_valid = StructureLoader(dataset_valid,
                                       batch_size=args.batch_size)

        reload_c = 0
        # Epoch loop
        for e in range(args.num_epochs):
            t0 = time.time()
            e = epoch + e  # Adjust epoch number if resuming from checkpoint
            model.train()  # Set model to training mode
            # Initialize epoch statistics
            train_sum, train_weights = 0., 0.
            train_acc = 0.
            # Reload training data periodically
            if e % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(
                        pdb_dict_train,
                        truncate=None,
                        max_length=args.max_protein_length)
                    loader_train = StructureLoader(dataset_train,
                                                   batch_size=args.batch_size)
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(
                        pdb_dict_valid,
                        truncate=None,
                        max_length=args.max_protein_length)
                    loader_valid = StructureLoader(dataset_valid,
                                                   batch_size=args.batch_size)
                    q.put_nowait(
                        executor.submit(get_pdbs, train_loader, 1,
                                        args.max_protein_length,
                                        args.num_examples_per_epoch))
                    p.put_nowait(
                        executor.submit(get_pdbs, valid_loader, 1,
                                        args.max_protein_length,
                                        args.num_examples_per_epoch))
                reload_c += 1
            # Training batch loop
            for _, batch in enumerate(loader_train):
                start_batch = time.time()
                # Featurize batch: convert protein structures to tensors
                X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                    batch, device)
                elapsed_featurize = time.time() - start_batch
                # Clear previous gradients
                optimizer.zero_grad()
                # Create loss mask: only compute loss for positions to be predicted
                mask_for_loss = mask * chain_M

                # Training step with optional mixed precision
                if args.mixed_precision:
                    # Forward pass with automatic mixed precision
                    with torch.cuda.amp.autocast():
                        log_probs = model(X, S, mask, chain_M, residue_idx,
                                          chain_encoding_all)
                        # Compute label-smoothed loss
                        _, loss_av_smoothed = loss_smoothed(
                            S, log_probs, mask_for_loss)

                    # Backward pass with gradient scaling
                    scaler.scale(loss_av_smoothed).backward()

                    # Optional gradient clipping
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)

                    # Update parameters with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard full-precision forward pass
                    log_probs = model(X, S, mask, chain_M, residue_idx,
                                      chain_encoding_all)
                    # Compute loss
                    _, loss_av_smoothed = loss_smoothed(
                        S, log_probs, mask_for_loss)
                    loss_av_smoothed.backward()

                    # Optional gradient clipping
                    if args.gradient_norm > 0.0:
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm)

                    # Update parameters
                    optimizer.step()

                # Compute accuracy metrics (not used for optimization, just logging)
                loss, loss_av, true_false = loss_nll(S, log_probs,
                                                     mask_for_loss)

                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false *
                                       mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()

                total_step += 1

            # Validation phase (no gradients needed)
            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0., 0.
                validation_acc = 0.
                # Validation batch loop
                for _, batch in enumerate(loader_valid):
                    # Featurize batch
                    X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(
                        batch, device)
                    # Forward pass
                    log_probs = model(X, S, mask, chain_M, residue_idx,
                                      chain_encoding_all)
                    mask_for_loss = mask * chain_M
                    # Compute loss and accuracy
                    loss, loss_av, true_false = loss_nll(
                        S, log_probs, mask_for_loss)

                    # Accumulate validation statistics
                    validation_sum += torch.sum(
                        loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(
                        true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(
                        mask_for_loss).cpu().data.numpy()

            # Compute epoch metrics
            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            # Format metrics for logging
            train_perplexity_ = np.format_float_positional(
                np.float32(train_perplexity), unique=False, precision=3)
            validation_perplexity_ = np.format_float_positional(
                np.float32(validation_perplexity), unique=False, precision=3)
            train_accuracy_ = np.format_float_positional(
                np.float32(train_accuracy), unique=False, precision=3)
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3)

            # Compute epoch time and log results
            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1 - t0),
                                            unique=False,
                                            precision=1)
            # Write to logfile
            with open(logfile, 'a') as f:
                f.write(
                    f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n'
                )
            # Also print to stdout
            print(
                f'epoch: {e+1}, step: {total_step}, time: {dt}, train: {train_perplexity_}, valid: {validation_perplexity_}, train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}'
            )

            # Save latest checkpoint (for resuming training)
            checkpoint_filename_last = base_folder + 'model_weights/epoch_last.pt'.format(
                e + 1, total_step)
            torch.save(
                {
                    'epoch': e + 1,
                    'step': total_step,
                    'num_edges': args.num_neighbors,
                    'noise_level': args.backbone_noise,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.optimizer.state_dict(),
                }, checkpoint_filename_last)

            # Save periodic checkpoints for model selection
            if (e + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = base_folder + 'model_weights/epoch{}_step{}.pt'.format(
                    e + 1, total_step)
                torch.save(
                    {
                        'epoch': e + 1,
                        'step': total_step,
                        'num_edges': args.num_neighbors,
                        'noise_level': args.backbone_noise,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':
                        optimizer.optimizer.state_dict(),
                    }, checkpoint_filename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data",
                           type=str,
                           default="my_path/pdb_2021aug02",
                           help="path for loading training data")
    argparser.add_argument("--path_for_outputs",
                           type=str,
                           default="./exp_020",
                           help="path for logs and model weights")
    argparser.add_argument(
        "--previous_checkpoint",
        type=str,
        default="",
        help="path for previous model weights, e.g. file.pt")
    argparser.add_argument("--num_epochs",
                           type=int,
                           default=200,
                           help="number of epochs to train for")
    argparser.add_argument("--save_model_every_n_epochs",
                           type=int,
                           default=10,
                           help="save model weights every n epochs")
    argparser.add_argument("--reload_data_every_n_epochs",
                           type=int,
                           default=2,
                           help="reload training data every n epochs")
    argparser.add_argument(
        "--num_examples_per_epoch",
        type=int,
        default=1000000,
        help="number of training example to load for one epoch")
    argparser.add_argument("--batch_size",
                           type=int,
                           default=10000,
                           help="number of tokens for one batch")
    argparser.add_argument("--max_protein_length",
                           type=int,
                           default=10000,
                           help="maximum length of the protein complext")
    argparser.add_argument("--hidden_dim",
                           type=int,
                           default=128,
                           help="hidden model dimension")
    argparser.add_argument("--num_encoder_layers",
                           type=int,
                           default=3,
                           help="number of encoder layers")
    argparser.add_argument("--num_decoder_layers",
                           type=int,
                           default=3,
                           help="number of decoder layers")
    argparser.add_argument("--num_neighbors",
                           type=int,
                           default=48,
                           help="number of neighbors for the sparse graph")
    argparser.add_argument("--dropout",
                           type=float,
                           default=0.1,
                           help="dropout level; 0.0 means no dropout")
    argparser.add_argument(
        "--backbone_noise",
        type=float,
        default=0.2,
        help="amount of noise added to backbone during training")
    argparser.add_argument("--rescut",
                           type=float,
                           default=3.5,
                           help="PDB resolution cutoff")
    argparser.add_argument("--debug",
                           type=bool,
                           default=False,
                           help="minimal data loading for debugging")
    argparser.add_argument(
        "--gradient_norm",
        type=float,
        default=-1.0,
        help="clip gradient norm, set to negative to omit clipping")
    argparser.add_argument("--mixed_precision",
                           type=bool,
                           default=True,
                           help="train with mixed precision")

    args = argparser.parse_args()
    main(args)
