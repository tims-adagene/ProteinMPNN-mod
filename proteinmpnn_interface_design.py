#!/usr/bin/env python
"""ProteinMPNN interface design script.

Standalone multi-chain sequence design using ProteinMPNN. Specify which chains
to redesign (designed) and which to hold fixed (fixed/visible), optionally
pin specific positions within designed chains, and apply per-position amino
acid omission or bias constraints.

Outputs FASTA files with designed sequences and scores.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import protein_mpnn_utils as mpnn_utils
from boring_utils.utils import tprint

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


# ---------- argument parsing ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ProteinMPNN multi-chain interface design."
    )
    # I/O
    parser.add_argument(
        "--pdb_path", type=str, default=None, help="Path to a single PDB file"
    )
    parser.add_argument(
        "--pdb_dir", type=str, default=None, help="Directory of PDB files to process"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Output directory for FASTA results (default: outputs)",
    )

    # Chain specification
    parser.add_argument(
        "--designed_chain",
        type=str,
        default="A",
        help="Chain(s) to redesign; comma/space separated (default: A)",
    )
    parser.add_argument(
        "--fixed_chain",
        type=str,
        default="",
        help="Chain(s) to keep fixed (provide context); comma/space separated",
    )

    # Position constraints
    parser.add_argument(
        "--fixed_positions",
        type=str,
        default=None,
        help="JSON: positions within designed chains to keep native. "
        'Format: {"chain": [pos1, pos2, ...]} (1-indexed). '
        "Example: '{\"A\": [1, 2, 3, 50, 51]}'",
    )
    parser.add_argument(
        "--position_omit_AA",
        type=str,
        default=None,
        help="JSON: per-position amino acids to exclude. "
        'Format: {"chain": {position: "AAs", ...}}. '
        'Example: \'{"A": {"1": "CW", "2": "C"}}\'',
    )
    parser.add_argument(
        "--position_bias_AA",
        type=str,
        default=None,
        help="JSON: per-position amino acid biases. "
        'Format: {"chain": {position: {"AA": bias, ...}, ...}}. '
        'Example: \'{"A": {"1": {"A": 0.5, "D": -1.0}}}\'',
    )

    # Model / inference controls
    parser.add_argument(
        "--model_name",
        type=str,
        default="v_48_020",
        help="Checkpoint stem name (default: v_48_020)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Directory containing <model_name>.pt",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit checkpoint path (overrides weights_dir)",
    )
    parser.add_argument(
        "--num_seq_per_target",
        type=int,
        default=1,
        help="Number of sequences to generate per structure (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch copies per sampling round (default: 1)",
    )
    parser.add_argument(
        "--sampling_temp",
        type=str,
        default="0.1",
        help="Space-separated sampling temperatures (default: 0.1)",
    )
    parser.add_argument(
        "--omit_AAs",
        type=str,
        default="X",
        help="Residue types to globally omit (default: X)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20000,
        help="Max length for featurization (default: 20000)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Model hidden dimension (default: 128)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of encoder/decoder layers (default: 3)",
    )
    return parser


# ---------- helpers ----------


def split_chains(chain_str: str) -> List[str]:
    """Parse 'A B C' or 'A,B,C' into ['A', 'B', 'C']."""
    if not chain_str:
        return []
    return [c for c in re.sub("[^A-Za-z]+", ",", chain_str).split(",") if c]


def resolve_checkpoint(args) -> Path:
    if args.checkpoint_path:
        ckpt = Path(args.checkpoint_path)
    else:
        weights_dir = (
            Path(args.weights_dir)
            if args.weights_dir
            else Path(mpnn_utils.__file__).resolve().parent / "vanilla_model_weights"
        )
        ckpt = weights_dir / f"{args.model_name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"ProteinMPNN checkpoint not found: {ckpt}")
    return ckpt


def load_model(
    checkpoint_path: Path, device: torch.device, hidden_dim: int, num_layers: int
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = mpnn_utils.ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=checkpoint.get("noise_level", 0.0),
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tprint(f"Model loaded from {checkpoint_path}")
    return model


def build_omit_AA_dict(
    position_omit_AA: Dict,
    pdb_name: str,
) -> Dict:
    """Convert user-friendly position_omit_AA to ProteinMPNN internal format.

    Input:  {"chain": {"pos": "AAs", ...}}   (positions are 1-indexed strings or ints)
    Output: {pdb_name: {chain: [[[positions], "AAs"], ...]}}
    """
    result: Dict = {pdb_name: {}}
    for chain, constraints in position_omit_AA.items():
        aa_to_positions: Dict[str, List[int]] = {}
        for pos, aas in constraints.items():
            aas_str = aas if isinstance(aas, str) else "".join(aas)
            aa_to_positions.setdefault(aas_str, []).append(int(pos))
        result[pdb_name][chain] = [
            [positions, aas_str] for aas_str, positions in aa_to_positions.items()
        ]
    return result


def build_bias_by_res_dict(
    position_bias_AA: Dict,
    pdb_name: str,
    chain_lengths: Dict[str, int],
) -> Dict:
    """Convert user-friendly position_bias_AA to ProteinMPNN internal format.

    Input:  {"chain": {"pos": {"AA": bias, ...}, ...}}
    Output: {pdb_name: {chain: np.ndarray[L, 21]}}
    """
    result: Dict = {pdb_name: {}}
    for chain, biases in position_bias_AA.items():
        chain_len = chain_lengths.get(chain, 0)
        if chain_len == 0:
            chain_len = max(int(p) for p in biases.keys()) if biases else 0
        bias_array = np.zeros([chain_len, len(ALPHABET)])
        for pos, aa_biases in biases.items():
            pos_idx = int(pos) - 1
            if 0 <= pos_idx < chain_len:
                for aa, val in aa_biases.items():
                    if aa in ALPHABET:
                        bias_array[pos_idx, ALPHABET.index(aa)] = float(val)
        result[pdb_name][chain] = bias_array
    return result


def get_chain_lengths(pdb_dict: dict) -> Dict[str, int]:
    """Extract {chain_letter: length} from a parsed PDB dict."""
    lengths: Dict[str, int] = {}
    for key in pdb_dict:
        if key.startswith("seq_chain_"):
            chain = key[len("seq_chain_") :]
            lengths[chain] = len(pdb_dict[key])
    return lengths


# ---------- main logic ----------


def design_one_pdb(
    pdb_path: str,
    model: torch.nn.Module,
    device: torch.device,
    args,
    designed_chain_list: List[str],
    fixed_chain_list: List[str],
    out_dir: Path,
):
    """Run ProteinMPNN design on a single PDB file."""
    chain_list = sorted(set(designed_chain_list + fixed_chain_list))
    pdb_dict_list = mpnn_utils.parse_PDB(pdb_path, input_chain_list=chain_list)
    if not pdb_dict_list:
        tprint(f"WARNING: No chains found in {pdb_path}, skipping.")
        return

    dataset = mpnn_utils.StructureDatasetPDB(
        pdb_dict_list, truncate=None, max_length=args.max_length
    )
    chain_id_dict = {pdb_dict_list[0]["name"]: (designed_chain_list, fixed_chain_list)}

    # Fixed positions within designed chains
    fixed_position_dict = None
    if args.fixed_positions:
        fp = json.loads(args.fixed_positions)
        fixed_position_dict = {pdb_dict_list[0]["name"]: fp}

    # Position-specific omit
    omit_AA_dict = None
    if args.position_omit_AA:
        omit_AA_dict = build_omit_AA_dict(
            json.loads(args.position_omit_AA),
            pdb_dict_list[0]["name"],
        )

    # Position-specific bias
    bias_by_res_dict = None
    if args.position_bias_AA:
        chain_lengths = get_chain_lengths(pdb_dict_list[0])
        bias_by_res_dict = build_bias_by_res_dict(
            json.loads(args.position_bias_AA),
            pdb_dict_list[0]["name"],
            chain_lengths,
        )

    temperatures = [float(t) for t in args.sampling_temp.split()]
    omit_AAs_np = np.array([AA in args.omit_AAs for AA in ALPHABET], dtype=np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))
    num_batches = max(1, int(np.ceil(args.num_seq_per_target / args.batch_size)))

    for protein in dataset:
        t0 = time.time()
        batch_clones = [copy.deepcopy(protein) for _ in range(args.batch_size)]

        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            chain_encoding_all,
            chain_list_list,
            visible_list_list,
            masked_list_list,
            masked_chain_length_list_list,
            chain_M_pos,
            omit_AA_mask,
            residue_idx,
            dihedral_mask,
            tied_pos_list_of_lists_list,
            pssm_coef,
            pssm_bias,
            pssm_log_odds_all,
            bias_by_res_all,
            tied_beta,
        ) = mpnn_utils.tied_featurize(
            batch_clones,
            device,
            chain_id_dict,
            fixed_position_dict=fixed_position_dict,
            omit_AA_dict=omit_AA_dict,
            tied_positions_dict=None,
            pssm_dict=None,
            bias_by_res_dict=bias_by_res_dict,
        )

        name_ = batch_clones[0]["name"]

        # Score native sequence
        randn_1 = torch.randn(chain_M.shape, device=device)
        log_probs_native = model(
            X,
            S,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn_1,
        )
        mask_for_loss = mask * chain_M * chain_M_pos
        native_scores = mpnn_utils._scores(S, log_probs_native, mask_for_loss)
        native_score = native_scores.cpu().data.numpy()

        # Collect results
        results: List[
            Tuple[str, float, float]
        ] = []  # (seq, sample_score, native_score)

        for temp in temperatures:
            for j in range(num_batches):
                randn_2 = torch.randn(chain_M.shape, device=device)
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
                    pssm_multi=0.0,
                    pssm_log_odds_flag=False,
                    pssm_log_odds_mask=(pssm_log_odds_all > 0.0).float(),
                    pssm_bias_flag=False,
                    bias_by_res=bias_by_res_all,
                )
                S_sample = sample_dict["S"]

                log_probs = model(
                    X,
                    S_sample,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn_2,
                    use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"],
                )
                scores = mpnn_utils._scores(
                    S_sample, log_probs, mask * chain_M * chain_M_pos
                )
                scores_np = scores.cpu().data.numpy()

                for b_ix in range(args.batch_size):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq = mpnn_utils._S_to_seq(S_sample[b_ix], chain_M[b_ix])

                    # Reorder sequence by chain letter
                    start, end = 0, 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end
                    seq_sorted = "".join(
                        list(np.array(list_of_AAs)[np.argsort(masked_list)])
                    )
                    # Insert '/' between chains
                    l0 = 0
                    for mc_length in list(
                        np.array(masked_chain_length_list)[np.argsort(masked_list)]
                    )[:-1]:
                        l0 += mc_length
                        seq_sorted = seq_sorted[:l0] + "/" + seq_sorted[l0:]
                        l0 += 1

                    results.append(
                        (seq_sorted, float(scores_np[b_ix]), float(native_score[b_ix]))
                    )
                    score_print = np.format_float_positional(
                        np.float32(scores_np[b_ix]), unique=False, precision=4
                    )
                    tprint(
                        f">T={temp}, sample={j * args.batch_size + b_ix}, score={score_print}\n{seq_sorted}"
                    )

        # Write FASTA output
        fa_path = out_dir / f"{name_}.fa"
        with open(fa_path, "w") as f:
            native_score_print = np.format_float_positional(
                np.float32(native_score[0]), unique=False, precision=4
            )
            f.write(f">native, score={native_score_print}\n")
            # Write native designed-chain sequence
            native_seq = mpnn_utils._S_to_seq(S[0], chain_M[0])
            f.write(f"{native_seq}\n")
            for idx, (seq, score, _) in enumerate(results):
                score_print = np.format_float_positional(
                    np.float32(score), unique=False, precision=4
                )
                f.write(f">sample_{idx}, score={score_print}\n{seq}\n")

        elapsed = int(time.time() - t0)
        tprint(
            f"Designed {len(results)} sequences for {name_} in {elapsed}s -> {fa_path}"
        )


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tprint(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint(args)
    model = load_model(checkpoint_path, device, args.hidden_dim, args.num_layers)

    designed_chain_list = split_chains(args.designed_chain)
    fixed_chain_list = split_chains(args.fixed_chain)

    if not designed_chain_list:
        sys.exit("ERROR: --designed_chain must specify at least one chain.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDB files
    pdb_paths: List[str] = []
    if args.pdb_path:
        pdb_paths.append(args.pdb_path)
    elif args.pdb_dir:
        pdb_paths = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    else:
        sys.exit("ERROR: Provide --pdb_path or --pdb_dir.")

    tprint(
        f"Processing {len(pdb_paths)} PDB file(s), designing chain(s) {designed_chain_list}"
    )

    for pdb_path in pdb_paths:
        try:
            design_one_pdb(
                pdb_path,
                model,
                device,
                args,
                designed_chain_list,
                fixed_chain_list,
                out_dir,
            )
        except KeyboardInterrupt:
            sys.exit("Interrupted by user.")
        except Exception as exc:
            tprint(f"ERROR processing {pdb_path}: {exc}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
