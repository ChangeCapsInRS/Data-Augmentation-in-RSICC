# This file combines the captions of the corresponding images in the given intent JSON files.
# This is mainly written to combine the captions of paraphrase augmentation results.

import argparse
import json
from pathlib import Path
from typing import Dict

from schemas import Captions


def merge_captions(
    main_intent_data: Dict, augmented_intent_dat: Dict, filename_suffix: str
) -> Dict:
    # create a dict with filename as key and the main caption as value
    main_captions = {
        intent["filename"]: intent for intent in main_intent_data["images"]
    }
    for intent in augmented_intent_dat["images"]:
        # get the filename
        filename: str = intent["filename"]
        filename_path = Path(intent["filename"])

        # remove the file extension
        assert filename.endswith(
            filename_suffix + filename_path.suffix
        ), f"Filename {filename} does not end with {filename_suffix}"

        original_filename = filename.replace(filename_suffix, "")

        # check if the filename is in the main captions
        assert (
            original_filename in main_captions
        ), f"Filename {original_filename} not found in main captions"

        # extend the sentences and sentids of the main captions
        main_captions[original_filename]["sentences"].extend(
            intent["sentences"]
        )
        main_captions[original_filename]["sentids"].extend(intent["sentids"])

    merged_captions = list(main_captions.values())

    # normalize the merged captions
    normalized_merged_captions = Captions.normalize_dict(
        {"images": merged_captions}
    )

    return normalized_merged_captions.model_dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Concatenate Captions",
        description="Combine the captions of the corresponding images in the given intent JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example usage: python src/combine_captions.py data/intent.json -a data/intent_augmented.json -o data/merged_captions.json",
    )
    parser.add_argument(
        "main_intent", type=str, help="Path to the main intent JSON file"
    )
    parser.add_argument(
        "--augmented_intent",
        "-a",
        type=str,
        help="Path to the augmented intent JSON file",
        required=True,
    )
    parser.add_argument(
        "--filename_suffix",
        "-s",
        type=str,
        default="_paraphrase",
        help="The suffixes at the end of filenames, indicating "
        'the type of augmentation. For example, "_paraphrase", "_rotate90" etc.',
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="merged_captions.json",
        help="Path to the output file",
    )
    args = parser.parse_args()

    main_intent = args.main_intent
    augmented_intent = args.augmented_intent
    filename_suffix = args.filename_suffix

    # Load the main intent JSON file
    with open(main_intent, "r") as f:
        main_intent_data: Captions = Captions.model_validate_json(f.read())

    # Load the augmented intent
    with open(augmented_intent, "r") as f:
        augmented_data: Captions = Captions.model_validate_json(f.read())

    # Merge the captions
    merged_captions = merge_captions(
        main_intent_data.model_dump(),
        augmented_data.model_dump(),
        filename_suffix,
    )

    # Save the merged captions
    with open(args.output, "w") as f:
        json.dump(merged_captions, f, indent=4)

    print(f"Merged captions saved to {args.output}")
