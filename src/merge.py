import argparse
import json
import os

from schemas import Captions


def set_up_parser():
    parser = argparse.ArgumentParser(
        prog="merge",
        description="Merges augmenter output JSON files and original captions "
        "JSON file to a single file to feed into the model",
    )

    parser.add_argument(
        "-d",
        "--input_directory_path",
        required=True,
        help="Path of the input directory that contains augmented JSON files",
    )
    parser.add_argument(
        "-o",
        "--output_file_path",
        required=True,
        help="Path of the generated output file",
    )

    args = parser.parse_args()
    return args


class Merger:
    file_extension = ".json"

    def __init__(self, args):
        self.args = args
        self.merged_images = []

    def add_file_string(self, file_path):
        with open(file_path, "r") as f:
            file_string = f.read()
            dictionary = json.loads(file_string)

            images = dictionary["images"]
            self.merged_images.extend(images)

    def merge(self):
        directory = os.fsencode(self.args.input_directory_path)

        for file in os.listdir(directory):
            file_name = os.fsdecode(file)

            if file_name.endswith(Merger.file_extension):
                file_path = os.path.join(
                    self.args.input_directory_path, file_name
                )
                self.add_file_string(file_path)

        # normalize the merged images
        normalized_captions = Captions.normalize_dict(
            {"images": self.merged_images}
        )

        self.merged_images = normalized_captions.model_dump()["images"]

    def write_merged_file(self):
        dictionary = {"images": self.merged_images}
        file_string = json.dumps(dictionary, indent=4)

        with open(self.args.output_file_path, "w") as f:
            print(file_string, file=f)


def main():
    args = set_up_parser()

    merger = Merger(args)
    merger.merge()
    merger.write_merged_file()


if __name__ == "__main__":
    main()
