import argparse
import json
import os
from copy import deepcopy
from typing import Dict

import augment
import schemas


def set_up_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="filter_captions",
        description="Filters out the captions for images that do not "
        "exist in the input images directory",
    )

    parser.add_argument(
        "--images_directory_path",
        required=True,
        help="Path of the directory that contains images",
    )
    parser.add_argument(
        "--captions_file_path",
        required=True,
        help="Path of the input JSON file that contains captions for images",
    )
    parser.add_argument(
        "--before_directory_name",
        required=True,
        help="Name of the input directory that contains before images",
    )
    parser.add_argument(
        "--output_file_path",
        required=True,
        help="Path of the output filtered JSON file",
    )

    args: argparse.Namespace = parser.parse_args()
    return args


class Filter:
    def __init__(self, args: argparse.Namespace):
        self.args: argparse.Namespace = args
        self.original_captions_string: str = None
        self.filtered_captions_string: str = None
        self.original_captions: schemas.Captions = None
        self.filtered_captions: schemas.Captions = None

    @staticmethod
    def image_exists(
        intent: schemas.Intent,
        images_directory_path: str,
        before_directory_name: str,
    ) -> bool:
        file_path: str = os.path.join(
            images_directory_path,
            intent.split,
            before_directory_name,
            f"{intent.split}_{intent.filename}",
        )
        is_file: bool = os.path.isfile(file_path)
        return is_file

    @staticmethod
    def apply(
        original_captions: schemas.Captions,
        images_directory_path: str,
        before_directory_name: str,
    ) -> schemas.Captions:
        filtered_captions: schemas.Captions = deepcopy(original_captions)

        filtered_captions.images = filter(
            lambda intent: Filter.image_exists(
                intent, images_directory_path, before_directory_name
            ),
            filtered_captions.images,
        )

        return filtered_captions

    @staticmethod
    def captions_to_string(captions: schemas.Captions) -> str:
        captions_dictionary: Dict = captions.model_dump()
        captions_string: str = json.dumps(captions_dictionary, indent=4)
        return captions_string

    @staticmethod
    def write_file(file_path: str, string: str):
        with open(file_path, "w") as f:
            print(string, file=f)


def main():
    args: argparse.Augmenter = set_up_parser()

    filter: Filter = Filter(args)
    filter.original_captions_string = augment.Augmenter.read_file(
        filter.args.captions_file_path
    )
    filter.original_captions = schemas.Captions.model_validate_json(
        filter.original_captions_string
    )
    print(f"Original caption count: {len(filter.original_captions.images)}")
    filter.filtered_captions = Filter.apply(
        filter.original_captions,
        filter.args.images_directory_path,
        filter.args.before_directory_name,
    )
    print(f"Filtered caption count: {len(filter.filtered_captions.images)}")
    filter.filtered_captions_string: str = Filter.captions_to_string(
        filter.filtered_captions
    )
    Filter.write_file(
        filter.args.output_file_path, filter.filtered_captions_string
    )


if __name__ == "__main__":
    main()
