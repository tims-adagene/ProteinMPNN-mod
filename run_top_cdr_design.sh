#!/bin/bash

# # Run antibody_demo_v2.py to get the top 15% CDR positions
# python antibody_demo_v2.py \
#     --pdb_path ./inputs/Antibody/6wgl.pdb \
#     --designed_chain H,L \
#     --fixed_chain T \
#     --cdr_ranges "H:26-35,50-66,99-114;L:24-39,55-61,94-102"

# Run single chain design_antibody_top_cdr.py to design the top 15% CDR positions
# python design_antibody_top_cdr.py \
#     --pdb_path ./inputs/Antibody/6wgl.pdb \
#     --designed_chain H \
#     --fixed_chain T,L \
#     --top_cdr_file ./outputs/6wgl/6wgl_chain_H_top15percent_cdr_scores.npy \
#     --out_folder ./outputs/6wgl_design/ \
#     --num_seqs 10 \
#     --temperature 0.1 \
#     --save_score 

# Run multi chain design_antibody_top_cdr.py to design the top 15% CDR positions
python design_antibody_top_cdr.py \
    --pdb_path ./inputs/Antibody/6wgl.pdb \
    --designed_chain H,L \
    --fixed_chain T \
    --top_cdr_files ./outputs/6wgl/6wgl_chain_H_top15percent_cdr_scores.npy ./outputs/6wgl/6wgl_chain_L_top15percent_cdr_scores.npy \
    --out_folder ./outputs/6wgl_design/ \
    --num_seqs 10 \
    --temperature 0.1 \
    --save_score 
