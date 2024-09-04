"""
This script is used to preprocess the captions in the JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil

from src.schemas import Captions


TEMPORARY_DIRECTORY = "temporary"
BEFORE_IMAGE_LABEL = "A"
AFTER_IMAGE_LABEL = "B"


def main():
    parser = argparse.ArgumentParser(description="Caption Preprocessing")

    parser.add_argument(
        "--captions",
        type=str,
        help="Path to the JSON file containing the captions",
    )
    parser.add_argument(
        "--images",
        type=str,
        help="Path to the images directory",
    )
    parser.add_argument(
        "--fixed_words",
        required=False,
        help="Path to the fixed words file",
    )
    parser.add_argument(
        "--train_amount",
        type=int,
        help="Percentage of train split",
    )
    parser.add_argument(
        "--validation_amount",
        type=int,
        help="Percentage of validation split",
    )
    parser.add_argument(
        "--test_amount",
        type=int,
        help="Percentage of test split",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--levircc",
        action="store_true",
        help="Use LEVIR-CC dataset",
    )

    arguments = parser.parse_args()

    assert (
        arguments.train_amount
        + arguments.validation_amount
        + arguments.test_amount
        == 100
    )

    # open and read the file
    # the captions are automatically preprocessed when the file is loaded
    captions = Captions.load(arguments.captions).model_dump()

    os.makedirs(arguments.output, exist_ok=True)
    if arguments.levircc:
        split_names = os.listdir(arguments.images)

        for image in captions["images"]:
            image["filename"] = (
                image["filename"]
                .replace("train_", "")
                .replace("val_", "")
                .replace("test_", "")
            )

        os.makedirs(TEMPORARY_DIRECTORY, exist_ok=True)
        image_ids = []
        filename_to_count = {}
        for split in split_names:
            split_path = os.path.join(arguments.images, split)
            for time_state in os.listdir(split_path):
                time_state_path = os.path.join(split_path, time_state)
                for file_name in os.listdir(time_state_path):
                    file_name_path = os.path.join(time_state_path, file_name)
                    image_id = (
                        file_name.replace("train_", "")
                        .replace("val_", "")
                        .replace("test_", "")[:-4]
                    )
                    if image_id in filename_to_count:
                        filename_to_count[image_id] += 1
                    else:
                        filename_to_count[image_id] = 1
                    suffix = (filename_to_count[image_id] - 1) // 2
                    unique_image_id = image_id
                    if suffix > 0:
                        unique_image_id += "_" + str(suffix)
                    for image in captions["images"]:
                        if (
                            image["filepath"] == split
                            and image["filename"][:-4] == image_id
                        ):
                            image["filename"] = unique_image_id + ".png"
                    image_ids.append(unique_image_id)
                    file_name_with_time_state = (
                        unique_image_id + "_" + time_state + file_name[-4:]
                    )
                    shutil.copyfile(
                        file_name_path,
                        os.path.join(
                            TEMPORARY_DIRECTORY,
                            file_name_with_time_state,
                        ),
                    )

        image_ids = list(set(image_ids))
        random.shuffle(image_ids)
        image_count = len(image_ids)
        train_split_size = int(image_count * arguments.train_amount / 100)
        validation_split_size = int(
            image_count * arguments.validation_amount / 100,
        )
        test_split_size = int(image_count * arguments.test_amount / 100)
        test_split_size += (
            image_count
            - train_split_size
            - validation_split_size
            - test_split_size
        )
        train_split_ids = image_ids[:train_split_size]
        validation_split_ids = image_ids[
            train_split_size: train_split_size + validation_split_size
        ]
        test_split_ids = image_ids[train_split_size + validation_split_size:]

        # Write image files from the temporary directory to the output
        # directory according to the determined split
        images_directory = os.path.join(
            arguments.output,
            os.path.basename(arguments.images),
        )
        os.makedirs(
            images_directory,
            exist_ok=True,
        )

        for split in split_names:
            os.makedirs(
                os.path.join(images_directory, split),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(images_directory, split, BEFORE_IMAGE_LABEL),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(images_directory, split, AFTER_IMAGE_LABEL),
                exist_ok=True,
            )

        for file_name in os.listdir(TEMPORARY_DIRECTORY):
            image_id = file_name[:-6]
            time_state = file_name[-5]
            if image_id in train_split_ids:
                split = "train"
            elif image_id in validation_split_ids:
                split = "val"
            elif image_id in test_split_ids:
                split = "test"
            else:
                raise ValueError("Invalid image id")
            output_file_name = file_name[:-6] + file_name[-4:]
            source_path = os.path.join(TEMPORARY_DIRECTORY, file_name)
            destination_path = os.path.join(
                arguments.images,
                split,
                time_state,
                output_file_name,
            )
            print(destination_path)
            shutil.copyfile(source_path, destination_path)

        # Splits in the captions file are no longer valid, correct them
        for image in captions["images"]:
            image_id = image["filename"][:-4]
            if image_id in train_split_ids:
                correct_split = "train"
            elif image_id in validation_split_ids:
                correct_split = "val"
            elif image_id in test_split_ids:
                correct_split = "test"
            else:
                raise ValueError("Invalid image id")
            image["filepath"] = correct_split
            image["split"] = correct_split

        shutil.copyfile(
            arguments.fixed_words,
            os.path.join(
                os.path.join(
                    arguments.output,
                    os.path.basename(
                        arguments.fixed_words,
                    ),
                ),
            ),
        )

    # write the validated content to the output file
    with open(
        os.path.join(
            arguments.output,
            os.path.basename(
                arguments.captions,
            ),
        ),
        "w",
    ) as f:
        f.write(json.dumps(captions, indent=4))


if __name__ == "__main__":
    main()
