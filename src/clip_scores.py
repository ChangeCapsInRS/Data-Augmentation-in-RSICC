import argparse
import json
from typing import List

import clip  # pip install -U git+https://github.com/openai/CLIP.git
import torch
from PIL import Image
from termcolor import cprint
from transformers import CLIPModel, CLIPProcessor

from src.schemas import Captions
from src.utils import convert_opencv_to_pillow_image


class ClipScorer:
    def __init__(self, fine_tuned: bool = True):
        self.model = None
        self.processor = None
        self.fine_tuned = fine_tuned
        self.device = None
        if self.fine_tuned is False:
            # Load the pre-trained CLIP model
            self.model, self.processor = clip.load("ViT-B/32")

            # Move the model to the GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
        else:
            self.model = CLIPModel.from_pretrained(
                "flax-community/clip-rsicd-v2"
            )
            self.processor = CLIPProcessor.from_pretrained(
                "flax-community/clip-rsicd-v2"
            )

    def get_clip_score(self, image: Image.Image, sentences: List[str]):
        if self.fine_tuned is True:
            inputs = self.processor(
                text=sentences, images=image, return_tensors="pt", padding=True  # type: ignore
            )

            outputs = self.model(**inputs)  # type: ignore
            logits_per_image = outputs.logits_per_image

            score = logits_per_image.mean() / 100.0
            score = score.item()
        else:
            # Preprocess the image and tokenize the text
            image_input = self.processor(image).unsqueeze(0)  # type: ignore
            text_input = clip.tokenize(sentences)

            # Move the inputs to GPU if available
            image_input = image_input.to(self.device)
            text_input = text_input.to(self.device)

            # Generate embeddings for the image and text
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)  # type: ignore
                text_features = self.model.encode_text(text_input)  # type: ignore

            # Normalize the features
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            # Calculate the cosine similarity with each sentence
            # and get the average score
            score = 0
            for i in range(len(sentences)):
                score += torch.matmul(
                    image_features,
                    text_features[i].permute(
                        *torch.arange(text_features[i].ndim - 1, -1, -1)
                    ),
                ).item()
            score = score / len(sentences)

        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="clip_scores", description="Computes CLIP scores"
    )
    parser.add_argument(
        "--fine_tuned", required=False, default=False, action="store_true"
    )
    parser.add_argument(
        "--images_path",
        required=False,
        default="",
    )
    parser.add_argument("--captions_path", required=False, default="")
    args = parser.parse_args()

    if args.fine_tuned is False:
        cprint("Using default CLIP score", "red")
    else:
        cprint("Using fine-tuned CLIP score", "red")

    image_path = args.images_path
    captions_path = args.captions_path
    with open(captions_path, "r") as file:
        captions = Captions.model_validate_json(file.read())

    # Get the image pairs with intent for the train, test, and val splits
    image_pairs = captions.to_ImagePairs(image_path, {"train", "test", "val"})

    # Get the CLIP score for each image pair
    scores = {}

    clip_scorer = ClipScorer(args.fine_tuned)

    for image_pair in image_pairs.pairs:
        # Combine the sentences into a single string
        # get the concatenated image
        combined_image = image_pair.get_concatenated_image()

        # convert the concatenated image to a PIL image
        pil_image = convert_opencv_to_pillow_image(combined_image)

        sentences = image_pair.intent.get_raw_sentences()

        score = clip_scorer.get_clip_score(pil_image, sentences)

        print(f"{image_pair.filename} CLIP Score: {score}")
        scores[image_pair.filename] = score

    with open("clip_scores.json", "w") as file:
        json.dump(scores, file)
