"""
This script is used to preprocess the captions in the JSON file.
"""

import argparse

from src.schemas import Captions

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Caption Preprocessing")

    # Add arguments
    parser.add_argument(
        "captions_path",
        type=str,
        help="Path to the JSON file containing the captions",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Path to the output JSON file",
    )
    # parser.add_argument(
    #     "--fix_errors",
    #     action="store_true",
    #     help="Fix the errors in the file if any (does not work currently)",
    #     default=False,
    # )

    # Parse the command-line arguments
    args = parser.parse_args()

    # open and read the file
    captions = Captions.load(args.captions_path)

    # dump the validated content to the output file
    with open(args.output_path, "w") as f:
        f.write(captions.model_dump_json())
