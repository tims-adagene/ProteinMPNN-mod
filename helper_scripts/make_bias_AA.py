"""
Script to create amino acid bias dictionary for ProteinMPNN

This script creates a dictionary that biases the model towards or away from specific amino acids
during sequence generation. Positive values make amino acids more likely, negative values make
them less likely.

Output format: {"AA": bias_value, ...}
Example: {"A": -0.01, "G": 0.02} - makes A less likely, G more likely

Usage:
    python make_bias_AA.py --output_path bias_dict.jsonl \\
                           --AA_list "A G F" \\
                           --bias_list "-0.01 0.02 0.5"
"""

import argparse


def main(args):
    """
    Main function to create amino acid bias dictionary

    Args:
        args: Command line arguments containing:
            - output_path: Path where the bias dictionary will be saved
            - bias_list: Space-separated list of bias values (floats)
            - AA_list: Space-separated list of amino acids (single letter codes)
    """
    import json

    # Convert bias values from string to list of floats
    bias_list = [float(item) for item in args.bias_list.split()]

    # Convert amino acid codes from string to list
    AA_list = [str(item) for item in args.AA_list.split()]

    # Create dictionary mapping amino acids to their bias values
    # Positive bias = more likely, negative bias = less likely
    my_dict = dict(zip(AA_list, bias_list))

    # Write the bias dictionary to output file in JSON format
    with open(args.output_path, 'w') as f:
        f.write(json.dumps(my_dict) + '\n')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--output_path",
                           type=str,
                           help="Path to save the output bias dictionary")
    argparser.add_argument("--AA_list",
                           type=str,
                           default='',
                           help="Space-separated list of amino acids to bias (e.g., 'A G F')")
    argparser.add_argument("--bias_list",
                           type=str,
                           default='',
                           help="Space-separated list of bias strengths (e.g., '-0.01 0.02 0.5')")

    args = argparser.parse_args()
    main(args)

# Example output:
# {"A": -0.01, "G": 0.02}
# This would make Alanine (A) less likely and Glycine (G) more likely in generated sequences
