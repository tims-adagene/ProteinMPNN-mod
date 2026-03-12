#!/usr/bin/env python
"""ProteinMPNN CDR confidence scorer.

This script scores native antibody sequences with ProteinMPNN and reports
per‑position negative log probabilities, highlighting CDR residues (user
provided) and the top 15% least confident CDR positions. Outputs mirror the
original utility but use the packaged rfantibody modules and a cleaner entry
point.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import protein_mpnn_utils as mpnn_utils
from boring_utils.utils import tprint

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


# ---------- argument parsing ----------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score native sequences with ProteinMPNN and report CDR confidence."
    )
    parser.add_argument("--pdb", type=str, default=None, help="PDB code to fetch from RCSB")
    parser.add_argument("--pdb_path", type=str, default=None, help="Local PDB path (takes priority over --pdb)")
    parser.add_argument("--designed_chain", type=str, default="C", help="Chain(s) to design; comma/space separated")
    parser.add_argument("--fixed_chain", type=str, default="A B", help="Chain(s) to keep fixed; comma/space separated")
    parser.add_argument(
        "--cdr_ranges",
        type=str,
        default="",
        help='Custom CDR ranges: "H:26-35,50-66,99-114;L:24-39,55-61,94-102"',
    )
    parser.add_argument("--out_folder", type=str, default=None, help="Output directory (default: ./outputs/temp)")
    parser.add_argument("--homomer", action="store_true", help="Treat chains as homomers (tie positions)")

    # model / inference controls
    parser.add_argument("--model_name", type=str, default="v_48_020", help="Checkpoint stem name")
    parser.add_argument("--weights_dir", type=str, default=None, help="Directory containing <model_name>.pt")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit checkpoint path (overrides weights_dir)")
    parser.add_argument("--sampling_temp", type=str, default="0.0001", help="Space separated sampling temperatures")
    parser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to sample per target")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for sampling")
    parser.add_argument("--omit_AAs", type=str, default="X", help="Residues to omit globally")
    parser.add_argument("--max_length", type=int, default=20000, help="Max length for featurization")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of encoder/decoder layers")
    parser.add_argument("--pssm_threshold", type=float, default=0.0, help="PSSM log-odds threshold")
    return parser


# ---------- helpers ----------

def parse_cdr_range(range_str: str) -> Dict[str, List[Tuple[int, int]]]:
    if not range_str:
        return {}
    result: Dict[str, List[Tuple[int, int]]] = {}
    for part in range_str.split(";"):
        if ":" not in part:
            continue
        chain, ranges = part.split(":", 1)
        chain = chain.strip()
        result[chain] = []
        for range_part in ranges.split(","):
            if "-" in range_part:
                start, end = map(int, range_part.split("-"))
                result[chain].append((start, end))
    return result


def split_chains(chain_str: str) -> List[str]:
    if not chain_str:
        return []
    return [c for c in re.sub("[^A-Za-z]+", ",", chain_str).split(",") if c]


def make_tied_positions_for_homomers(pdb_dict_list: Iterable[dict]) -> Dict[str, List[Dict[str, List[int]]]]:
    homomer_dict: Dict[str, List[Dict[str, List[int]]]] = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in result if item.startswith("seq_chain")])
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        tied_positions_list: List[Dict[str, List[int]]] = []
        for pos in range(1, chain_length + 1):
            tied_positions_list.append({chain: [pos] for chain in all_chain_list})
        homomer_dict[result["name"]] = tied_positions_list
    return homomer_dict


def resolve_checkpoint(args) -> Path:
    if args.checkpoint_path:
        ckpt = Path(args.checkpoint_path)
    else:
        weights_dir = Path(args.weights_dir) if args.weights_dir else Path(mpnn_utils.__file__).resolve().parent / "vanilla_model_weights"
        ckpt = weights_dir / f"{args.model_name}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"ProteinMPNN checkpoint not found: {ckpt}")
    return ckpt


def load_model(checkpoint_path: Path, device: torch.device, hidden_dim: int, num_layers: int) -> torch.nn.Module:
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
    if "noise_level" in checkpoint:
        tprint(f"Training noise level: {checkpoint['noise_level']} Å")
    return model


def save_cdr_outputs(
    name_: str,
    designed_chain_list: List[str],
    masked_chain_length_list_list,
    masked_list_list,
    native_loss_np: np.ndarray,
    mask_for_loss_np: np.ndarray,
    cdr_ranges: Dict[str, List[Tuple[int, int]]],
    out_folder: Path,
):
    for b_ix, masked_chain_length_list in enumerate(masked_chain_length_list_list):
        masked_list = masked_list_list[b_ix]
        for designed_chain in designed_chain_list:
            chain_idx_list = [i for i, chain in enumerate(masked_list) if chain == designed_chain]
            if not chain_idx_list:
                continue
            chain_idx = chain_idx_list[0]
            start_idx = sum(masked_chain_length_list[:chain_idx])
            end_idx = start_idx + masked_chain_length_list[chain_idx]
            chain_losses = native_loss_np[b_ix, start_idx:end_idx]

            cdr_positions: List[int] = []
            cdr_scores: List[float] = []
            for pos, score in enumerate(chain_losses):
                actual_pos = pos + 1
                if mask_for_loss_np[b_ix, start_idx + pos] > 0 and is_in_cdr_range(designed_chain, actual_pos, cdr_ranges):
                    cdr_positions.append(actual_pos)
                    cdr_scores.append(float(score))

            chain_scores_output_file = out_folder / f"{name_}_chain_{designed_chain}_position_scores.npy"
            np.save(chain_scores_output_file, chain_losses)

            cdr_scores_output_file = out_folder / f"{name_}_chain_{designed_chain}_cdr_position_scores.npy"
            np.save(cdr_scores_output_file, np.array([cdr_positions, cdr_scores], dtype=object))

            if not cdr_scores:
                continue

            score_pos_pairs = sorted(zip(cdr_scores, cdr_positions), reverse=True)
            top_n = max(1, int(len(score_pos_pairs) * 0.15))
            top_positions = [pos for _, pos in score_pos_pairs[:top_n]]
            top_scores = [score for score, _ in score_pos_pairs[:top_n]]

            top_cdr_scores_output_file = out_folder / f"{name_}_chain_{designed_chain}_top15percent_cdr_scores.npy"
            np.save(top_cdr_scores_output_file, np.array([top_positions, top_scores], dtype=object))

            try:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(range(len(cdr_positions)), cdr_scores, alpha=0.7)
                plt.xticks(range(len(cdr_positions)), cdr_positions, rotation=90)
                plt.xlabel("CDR Position")
                plt.ylabel("-log P(native AA)")
                plt.title(f"Chain {designed_chain} CDR position scores")
                top_indices = [cdr_positions.index(pos) for pos in top_positions if pos in cdr_positions]
                for idx in top_indices:
                    bars[idx].set_color("red")
                    bars[idx].set_alpha(1.0)
                from matplotlib.patches import Patch

                plt.legend(handles=[Patch(facecolor="red", alpha=1.0, label=f"Top {top_n} positions"), Patch(facecolor="blue", alpha=0.7, label="Other CDR positions")])
                plot_path = out_folder / f"{name_}_chain_{designed_chain}_cdr_scores.png"
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
            except Exception as exc:  # plotting is best-effort
                tprint(f"Plotting failed for chain {designed_chain}: {exc}")


def is_in_cdr_range(chain: str, position: int, custom_ranges: Dict[str, List[Tuple[int, int]]] | None) -> bool:
    if custom_ranges and chain in custom_ranges:
        for start, end in custom_ranges[chain]:
            if start <= position <= end:
                return True
    return False


def run(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tprint(f"Using device: {device}")

    pdb_path = args.pdb_path if args.pdb_path else mpnn_utils.get_pdb(args.pdb)
    out_folder = Path(args.out_folder or "./outputs/temp")
    out_folder.mkdir(parents=True, exist_ok=True)

    cdr_ranges = parse_cdr_range(args.cdr_ranges)
    designed_chain_list = split_chains(args.designed_chain)
    fixed_chain_list = split_chains(args.fixed_chain)
    chain_list = sorted(set(designed_chain_list + fixed_chain_list))

    pdb_dict_list = mpnn_utils.parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = mpnn_utils.StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    chain_id_dict = {pdb_dict_list[0]["name"]: (designed_chain_list, fixed_chain_list)}
    tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list) if args.homomer else None

    checkpoint_path = resolve_checkpoint(args)
    model = load_model(checkpoint_path, device, args.hidden_dim, args.num_layers)

    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_np = np.array([AA in args.omit_AAs for AA in ALPHABET], dtype=np.float32)
    bias_AAs_np = np.zeros(len(ALPHABET))
    num_batches = max(1, int(np.ceil(args.num_seq_per_target / args.batch_size)))

    for protein in dataset_valid:
        score_list: List[float] = []
        loss_list: List[np.ndarray] = []
        all_probs_list: List[np.ndarray] = []
        all_log_probs_list: List[np.ndarray] = []
        S_sample_list: List[np.ndarray] = []

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
            fixed_position_dict=None,
            omit_AA_dict=None,
            tied_positions_dict=tied_positions_dict,
            pssm_dict=None,
            bias_by_res_dict=None,
        )

        pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float()
        name_ = batch_clones[0]["name"]

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
        native_scores, native_losses = mpnn_utils._scores_w_loss(S, log_probs_native, mask_for_loss)
        native_score_np = native_scores.cpu().data.numpy()
        native_loss_np = native_losses.cpu().data.numpy()
        mask_for_loss_np = mask_for_loss.cpu().data.numpy()

        for temp in temperatures:
            for j in range(num_batches):
                randn_2 = torch.randn(chain_M.shape, device=device)
                if tied_positions_dict is None:
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
                        pssm_log_odds_flag=bool(args.pssm_threshold),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=False,
                        bias_by_res=bias_by_res_all,
                    )
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
                        pssm_multi=0.0,
                        pssm_log_odds_flag=bool(args.pssm_threshold),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=False,
                        tied_pos=tied_pos_list_of_lists_list[0],
                        tied_beta=tied_beta,
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
                mask_for_loss_sample = mask * chain_M * chain_M_pos
                scores, losses = mpnn_utils._scores_w_loss(S_sample, log_probs, mask_for_loss_sample)

                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())

                scores_np = scores.cpu().data.numpy()
                losses_np = losses.cpu().data.numpy()

                for b_ix in range(args.batch_size):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq = mpnn_utils._S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    score_list.append(scores_np[b_ix])
                    loss_list.append(losses_np[b_ix])

                    # Human-readable sequence formatting
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end
                    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(np.array(masked_chain_length_list)[np.argsort(masked_list)])[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + "/" + seq[l0:]
                        l0 += 1

                    score_print = np.format_float_positional(np.float32(scores_np[b_ix]), unique=False, precision=4)
                    tprint(f">T={temp}, sample={b_ix}, score={score_print}\n{seq}\n")

        all_probs_concat = np.concatenate(all_probs_list) if all_probs_list else np.array([])
        all_log_probs_concat = np.concatenate(all_log_probs_list) if all_log_probs_list else np.array([])
        S_sample_concat = np.concatenate(S_sample_list) if S_sample_list else np.array([])

        save_cdr_outputs(
            name_,
            designed_chain_list,
            masked_chain_length_list_list,
            masked_list_list,
            native_loss_np,
            mask_for_loss_np,
            cdr_ranges,
            out_folder,
        )

        probs_output_file = out_folder / f"{name_}_probs.npy"
        np.save(probs_output_file, all_probs_concat)

        log_probs_output_file = out_folder / f"{name_}_log_probs.npy"
        np.save(log_probs_output_file, all_log_probs_concat)

        scores_output_file = out_folder / f"{name_}_scores.npy"
        np.save(scores_output_file, np.array(score_list))

        native_score_output_file = out_folder / f"{name_}_native_score.npy"
        np.save(native_score_output_file, native_score_np)

        for designed_chain in designed_chain_list:
            chain_scores_dict = {}
            for b_ix in range(args.batch_size):
                masked_chain_length_list = masked_chain_length_list_list[b_ix]
                masked_list = masked_list_list[b_ix]
                chain_idx_list = [i for i, chain in enumerate(masked_list) if chain == designed_chain]
                if chain_idx_list:
                    chain_idx = chain_idx_list[0]
                    start_idx = sum(masked_chain_length_list[:chain_idx])
                    end_idx = start_idx + masked_chain_length_list[chain_idx]
                    chain_scores_dict[f"batch_{b_ix}"] = {
                        "scores": native_loss_np[b_ix, start_idx:end_idx]
                    }
            if chain_scores_dict:
                chain_scores_file = out_folder / f"{name_}_chain_{designed_chain}_all_scores.npz"
                np.savez(chain_scores_file, **chain_scores_dict)

        tprint(f"Completed scoring for {name_}; outputs in {out_folder}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
