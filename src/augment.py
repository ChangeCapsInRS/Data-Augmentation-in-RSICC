"""
Augment the images and captions using the selected augmenters.
The augmented images and captions are saved in the output folder.
The original images and captions are also copied to the output folder.
The augmented JSON files are merged into a single JSON file.
"""

import argparse
import os
import random
import shutil
import sys
from argparse import Namespace
from typing import Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation_methods import METHOD_NAME_TO_FUNCTION  # noqa: E402
from merge import Merger  # noqa: E402
from schemas import Captions  # noqa: E402

RANDOM_SEED = 42


def get_inputs(
    images_directory_name: str,
    json_path: str,
    splits: set[str],
    levircc: bool = False,
):
    captions = Captions.load(json_path)

    vocab, _ = captions.get_vocabulary()

    # convert model to image pairs and captions
    pydantic_pairs = captions.to_ImagePairs(
        images_directory_name, splits, levircc=levircc
    )

    return pydantic_pairs, vocab


def get_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images-directory",
        "-i",
        type=str,
        default=r"data/SampleImagesWithoutAugmentation",
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--json-path",
        "-j",
        type=str,
        default=r"data/merged_no_aug_without_unicode_sampled.json",
        help="Path to the JSON file containing the captions",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        required=True,
        help="Path to the output folder",
    )
    parser.add_argument(
        "--augmenter-names",
        "-a",
        required=True,
        nargs="+",
        help="List of augmenters to run",
        choices=METHOD_NAME_TO_FUNCTION.keys(),
    )
    parser.add_argument(
        "--split",
        "-s",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="List of splits to augment",
    )
    parser.add_argument(
        "--overwrite",
        "-w",
        action="store_true",
        help="Overwrite the output files if they already exist",
    )
    parser.add_argument(
        "--make-directories",
        "-m",
        help="Create the output directories if they do not exist",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--levircc",
        action="store_true",
        help="Use LEVIR-CC dataset",
    )
    args = parser.parse_args(argv)

    return args


def merge_files(output_folder):
    args_dictionary = {
        "input_directory_path": output_folder,
        "output_file_path": os.path.join(output_folder, "merged.json"),
    }
    merger_args = Namespace(**args_dictionary)

    merger = Merger(merger_args)
    merger.merge()
    merger.write_merged_file()


def copy_original_files(images_directory, json_path, output_folder):
    shutil.copy(json_path, os.path.join(output_folder, "original.json"))
    shutil.copytree(
        images_directory,
        output_folder,
        dirs_exist_ok=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    # set random seed
    random.seed(RANDOM_SEED)

    args = get_args(argv)
    print(args)
    selected_methods = [
        METHOD_NAME_TO_FUNCTION[method_name]
        for method_name in args.augmenter_names
    ]
    print("Reading input...")
    pairs, vocab = get_inputs(
        args.images_directory,
        args.json_path,
        set(args.split),
        levircc=args.levircc,
    )

    print(f"Augmenting images and captions using {args.augmenter_names}...")
    for augmenter_function in selected_methods:
        augmentation_result = pairs.augment(
            augmenter_function,
            vocab=vocab,
        )
        augmentation_result.save(
            output_path=args.output_folder,
            overwrite=args.overwrite,
            make_dirs=args.make_directories,
            dataset="LevirCC" if args.levircc else "SecondCC",
        )

    # copy original files to output folder
    print("Copying original files to output folder...")
    copy_original_files(
        args.images_directory, args.json_path, args.output_folder
    )

    # merge the augmented json files
    print("Merging augmented json files...")
    merge_files(args.output_folder)

    print("Augmentation is completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
