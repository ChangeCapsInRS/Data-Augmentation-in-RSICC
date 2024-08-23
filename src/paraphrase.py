# TODO: eliminate paraphrased sentences with low CLIP score
# TODO: add example usage in the argparse help message
# TODO: add type hints to the main function
# TODO: add shorter versions of the arguments to the argparse


import argparse
import copy
import functools
import json
import os
import random
import tempfile
from time import sleep
from typing import List, Literal, Optional, Sequence, Union

import cv2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from google.api_core.exceptions import (
    Aborted,
    AlreadyExists,
    BadGateway,
    BadRequest,
    Cancelled,
    ClientError,
    Conflict,
    DataLoss,
    DeadlineExceeded,
    DuplicateCredentialArgs,
    FailedPrecondition,
    Forbidden,
    GatewayTimeout,
    InternalServerError,
    InvalidArgument,
    PermissionDenied,
    PreconditionFailed,
    Redirection,
    ResourceExhausted,
    RetryError,
    ServerError,
    ServiceUnavailable,
    TemporaryRedirect,
    TooManyRequests,
    Unauthenticated,
    Unauthorized,
    Unknown,
)
from PIL import Image
from tqdm import tqdm

from src.clip_scores import ClipScorer
from src.schemas import Captions
from src.utils import (
    check_alpha_numeric,
    check_case,
    check_for_banned_words,
    check_sentence_length,
    check_sentence_vocab,
    convert_opencv_to_pillow_image,
    get_similar_sentences,
    normalize_sentence,
    remove_extra_spaces,
    remove_punctuation,
)

# add your API key to the .env file
# GENAI_API_KEY=your_api_key


@functools.lru_cache(maxsize=1)  # cache the model
def load_model() -> genai.GenerativeModel:
    # load the API key from the .env file
    load_dotenv()
    api_key = os.getenv("GENAI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    return model


# Default values for the arguments
default_lower_word_count_bound: int = 3
default_upper_word_count_bound: int = 15
default_reattempt_count: int = 5
default_similarity_threshold: float = 0.45
default_similarity_metric: Literal["bleu", "meteor"] = "bleu"
default_paraphrase_candidate_count: int = 3
default_verbose: bool = False
default_no_clip_score: bool = False
default_force_vocab: bool = False
default_wait_seconds: float = 1


def __generate_payload(
    sentence: str,
    vocab: set[str],
    lower_word_count_bound: int = default_lower_word_count_bound,
    upper_word_count_bound: int = default_upper_word_count_bound,
    errors: List[str] = list(),
):
    prompt = (
        "You will be given a sentence and a vocabulary. "
        "The given sentence explains the changes in a before and after picture, which has not been provided to you. "
        "Paraphrase the sentence so that the changes described in the original sentence are **preserved**. "
        "**DO NOT CHANGE THE MEANING OF THE SENTENCE OR "
        "ADD ANY NEW INFORMATION THAT IS NOT PRESENT IN THE ORIGINAL SENTENCE. **"
        "Imitate the style and context of the sentence. "
        f"Keep the length of the sentence between {lower_word_count_bound}-{upper_word_count_bound} words. "
        # "The length of the sentence should be similar to the original sentence. "
        "Return only the paraphrased phrase, do not provide any additional information. "
        "Do not mention the original sentence or the paraphrasing process in your answer. "
        "The paraphrased sentence should express a final and complete thought. "
        "The generated sentence should not be a fragment or a continuation of the original sentence. "
        "The paraphrased sentence should use only the words given in the vocabulary. \n\n"
        # The model does not enforce the use of these words in the vocabulary,
        # This is only a suggestion.
        f"**Vocabulary**: {vocab}\n\n"
        f"**Sentence**: {sentence}\n\n"
    )

    # add error message to the prompt
    if errors:
        prompt += (
            "This is not the first attempt at paraphrasing. "
            "The following errors were encountered during the previous attempts. "
            "Generate the paraphrased sentences, with the errors in mind.\n"
            f"**Errors**: {errors}\n\n"
        )

    payload = [
        prompt,
    ]

    return payload


def __get_response(payload):
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    while True:
        try:
            model = load_model()
            response = model.generate_content(
                payload, safety_settings=safety_settings
            )
            break
        except DeadlineExceeded:
            # retry if the request times out
            print("Request timed out. Retrying...")
            sleep(5)
        except TooManyRequests:
            # retry if the request limit is reached
            print("Request limit reached. Waiting for 5 minutes...")

            # add visual timer
            for _ in tqdm(range(300), desc="Waiting", unit="s"):
                sleep(1)
        except (
            BadGateway,
            GatewayTimeout,
            InternalServerError,
            ServiceUnavailable,
            DataLoss,
            Unknown,
            Aborted,
            Cancelled,
            FailedPrecondition,
            ResourceExhausted,
            Unauthenticated,
            PermissionDenied,
            ResourceExhausted,
            AlreadyExists,
            BadRequest,
            ClientError,
            Conflict,
            InvalidArgument,
            ServerError,
            TemporaryRedirect,
            Redirection,
            Forbidden,
            PreconditionFailed,
            Unauthorized,
            DuplicateCredentialArgs,
            RetryError,
        ) as e:
            # retry if there is a server error
            print(f"Encountered server error: {e}. Retrying...")
            sleep(5)

    return response


def __check_constraints(
    paraphrased_sentence: str,
    vocab: set[str],
    lower_word_count_bound: int,
    upper_word_count_bound: int,
    force_vocab: bool,
):
    check_case(paraphrased_sentence)
    check_alpha_numeric(paraphrased_sentence)
    check_sentence_length(
        paraphrased_sentence,
        lower_word_count_bound,
        upper_word_count_bound,
    )

    check_for_banned_words(paraphrased_sentence)
    if force_vocab:
        check_sentence_vocab(paraphrased_sentence, vocab)

    return paraphrased_sentence


def __paraphrase(
    sentence: str,
    *,
    vocab: set[str],
    lower_word_count_bound: int = default_lower_word_count_bound,
    upper_word_count_bound: int = default_upper_word_count_bound,
    force_vocab: bool = default_force_vocab,
    reattempt_count: int = default_reattempt_count,
    errors: List[
        str
    ] = list(),  # list of errors encountered during previous attempts
):
    assert (
        lower_word_count_bound > 0
    ), "The lower word count bound must be greater than 0."
    assert (
        upper_word_count_bound >= lower_word_count_bound
    ), "The upper word count bound must be greater than or equal to the lower word count bound."
    assert (
        reattempt_count >= 0
    ), "The number of reattempts must be greater than or equal to 0."
    assert len(sentence) > 0, "The sentence must not be empty."
    assert len(vocab) > 0, "The vocabulary must not be empty."

    # generate the payload
    payload = __generate_payload(
        sentence,
        vocab,
        lower_word_count_bound,
        upper_word_count_bound,
        errors,
    )

    # get the response from the model
    response = __get_response(payload)

    try:
        # get the response text
        response_text = response.text

        # normalize the paraphrased sentence
        normalized_paraphrased_sentence = normalize_sentence(response_text)

        # check if the sentence violates any of the constraints
        __check_constraints(
            normalized_paraphrased_sentence,
            vocab,
            lower_word_count_bound,
            upper_word_count_bound,
            force_vocab,
        )
    except ValueError as e:
        # print the error message
        print(
            f'\nEncountered error while paraphrasing "{sentence}": {e.args[0]}'
        )

        # add the error message to the list of errors
        new_errors = errors.copy()
        new_errors.append(e.args[0])
        if reattempt_count > 0:
            # reattempt to paraphrase the sentence with the new errors
            print(
                f"Reattempting to paraphrase the sentence. Attempts left: {reattempt_count}"
            )
            return __paraphrase(
                sentence,
                vocab=vocab,
                lower_word_count_bound=lower_word_count_bound,
                upper_word_count_bound=upper_word_count_bound,
                force_vocab=force_vocab,
                reattempt_count=reattempt_count - 1,
                errors=new_errors,
            )
        else:
            # reraise the error if the maximum number of reattempts have been reached
            raise ValueError(
                f"Maximum number of reattempts reached. {e.args[0]}"
            )

    # return the normalized paraphrased sentence if it passes all the constraints
    return normalized_paraphrased_sentence


def paraphrase(
    sentence: str,
    *,
    vocab: set[str],
    lower_word_count_bound: int = default_lower_word_count_bound,
    upper_word_count_bound: int = default_upper_word_count_bound,
    force_vocab: bool = default_force_vocab,
    reattempt_count: int = default_reattempt_count,
    paraphrase_candidate_count: int = default_paraphrase_candidate_count,
) -> List[str]:
    """Paraphrase a sentence using the Gemini Pro Vision model.

    Args:
        sentence (str): The sentence to paraphrase.
        vocab (set[str]): The vocabulary to use for paraphrasing.
        lower_word_count_bound (int, optional): The lower word count bound for the paraphrased sentence.
        upper_word_count_bound (int, optional): The upper word count bound for the paraphrased sentence.
        force_vocab (bool, optional): Whether to force the model to only use words from the vocabulary. The returned sentence may contain words outside the vocabulary, if force_vocab is False.
        reattempt_count (int, optional): The number of reattempts to paraphrase the sentence.
        paraphrase_candidate_count (int, optional): The number of paraphrased sentences to generate
        for each sentence.

    Returns:
        List[str]: A list of paraphrased sentences. Each sentence is a paraphrased version of the input sentence.
        The length of the list may be less than the paraphrase_candidate_count if some candidates could not be generated.
    """
    candidates = []
    for candidate_no in range(paraphrase_candidate_count):
        try:
            paraphrased_sentence = __paraphrase(
                sentence,
                vocab=vocab,
                lower_word_count_bound=lower_word_count_bound,
                upper_word_count_bound=upper_word_count_bound,
                force_vocab=force_vocab,
                reattempt_count=reattempt_count,
            )
            candidates.append(paraphrased_sentence)
        except ValueError as e:
            # max number of reattempts have been reached and the sentence could not be paraphrased
            print(
                f'Encountered error while generating {candidate_no+1}th paraphrased candidate for "{sentence}": ',
                e.args[0],
            )

    return candidates


def __remove_similar_sentences(
    original_sentences: List[str],
    paraphrased_sentences: List[List[str]],
    similarity_threshold: float,
    similarity_metric: Literal["meteor", "bleu"],
    verbose: bool = default_verbose,
) -> List[List[str]]:
    # make a copy of the paraphrased sentences
    result_paraphrased_sentences = copy.deepcopy(paraphrased_sentences)

    # eliminate paraphrased sentences that are too similar to the other sentences
    for i, paraphrase_candidates in enumerate(result_paraphrased_sentences):
        # skip empty lists
        if not paraphrase_candidates:
            continue

        for j, paraphrased_sentence in enumerate(paraphrase_candidates):
            other_sentences = (
                original_sentences
                + sum(paraphrased_sentences[:i], [])
                + paraphrase_candidates[:j]
            )
            similar_sentences = get_similar_sentences(
                paraphrased_sentence,
                other_sentences,
                similarity_threshold,
                similarity_metric,
            )

            # remove the paraphrased sentence if it is too similar to any of the previous sentences
            if similar_sentences:
                if verbose:
                    # print the similar sentences
                    print(
                        f'Paraphrased sentence, "{paraphrased_sentence}" is similar to the following sentence(s): '
                    )
                    for index, sentence, score in similar_sentences:
                        print(
                            f'"{sentence}" with a {similarity_metric.upper()} score of {score:.2f}'
                        )

                    print("Removing the paraphrased sentence.")

                # remove the paraphrased sentence
                result_paraphrased_sentences[i][j] = ""

    # remove empty strings from the list
    result_paraphrased_sentences = [
        [sentence for sentence in sentences if sentence]
        for sentences in result_paraphrased_sentences
    ]

    return result_paraphrased_sentences


def pick_best_candidates(
    paraphrase_candidates: List[List[str]],
    concat_image: Image.Image,
):
    # get the CLIP score for each candidate
    clip_scorer = ClipScorer()
    clip_scores = {
        candidate: clip_scorer.get_clip_score(concat_image, [candidate])
        for candidates in paraphrase_candidates
        for candidate in candidates
    }

    chosen_candidates: List[Union[None, str]] = [None] * len(
        paraphrase_candidates
    )  # the chosen candidate for each sentence
    remaining_candidates = copy.deepcopy(
        paraphrase_candidates
    )  # the remaining candidates for each sentence
    empty_candidates = []  # the indices of the sentences with no candidates

    # choose the best candidate for each sentence
    for i, candidates in enumerate(paraphrase_candidates):
        # if there are no candidates for the sentence
        if not candidates:
            empty_candidates.append(i)
            continue

        # select the candidate with the highest CLIP score
        chosen_candidate_index = np.argmax(
            [clip_scores[candidate] for candidate in candidates]
        )
        chosen_candidates[i] = candidates[chosen_candidate_index]

        # remove the chosen candidate from the list of remaining candidates
        del remaining_candidates[i][chosen_candidate_index]

    if len(empty_candidates) == len(paraphrase_candidates):
        raise ValueError("No candidates to choose from.")

    # if there are empty candidates
    if empty_candidates:
        # flatten the list of remaining candidates
        remaining_candidates_flat = [
            candidate
            for sublist in remaining_candidates
            for candidate in sublist
        ]
        filler_candidates = []  # the candidates to fill the empty candidates

        # if there are not enough remaining candidates to fill the empty candidates
        if len(remaining_candidates_flat) < len(empty_candidates):
            # duplicate the already chosen or remaining candidates
            duplicate_candidate_amount = len(empty_candidates) - len(
                remaining_candidates_flat
            )
            choices = remaining_candidates_flat + [
                candidate for candidate in chosen_candidates if candidate
            ]
            weights = np.array([clip_scores[c] for c in choices])
            weights /= weights.sum()  # normalize the weights

            # add the remaining candidates to the filler candidates
            filler_candidates.extend(remaining_candidates_flat)
            # randomly select candidates based on the CLIP score
            filler_candidates.extend(
                random.choices(
                    choices, k=duplicate_candidate_amount, weights=weights
                )
            )

        # if there are more remaining candidates than empty candidates
        elif len(remaining_candidates_flat) > len(empty_candidates):
            # select the candidates with the highest CLIP score
            filler_candidates = sorted(
                remaining_candidates_flat,
                key=lambda x: clip_scores[x],
                reverse=True,
            )[: len(empty_candidates)]
        else:
            filler_candidates.extend(remaining_candidates_flat)

        # sanity check
        assert len(filler_candidates) == len(empty_candidates), (
            "The number of remaining candidates must be equal to the number of empty candidates. "
            f"{len(filler_candidates)=}, {len(empty_candidates)=}"
        )

        # fill the empty candidates with the remaining candidates
        for i, candidate in zip(empty_candidates, filler_candidates):
            chosen_candidates[i] = candidate

    result: List[str] = [
        candidate for candidate in chosen_candidates if candidate
    ]
    assert len(result) == len(paraphrase_candidates), (
        "The number of chosen candidates must be equal to the number of paraphrased candidates. "
        f"{len(result)=}, {len(paraphrase_candidates)=}"
    )

    return result


def paraphrase_intent(
    original_sentences: Sequence[str],
    *,
    vocab: set[str],
    before_image: np.ndarray,
    after_image: np.ndarray,
    lower_word_count_bound: int = default_lower_word_count_bound,
    upper_word_count_bound: int = default_upper_word_count_bound,
    force_vocab: bool = default_force_vocab,
    reattempt_count: int = default_reattempt_count,
    similarity_threshold: float = default_similarity_threshold,
    similarity_metric: Literal["meteor", "bleu"] = default_similarity_metric,
    paraphrase_candidate_count: int = default_paraphrase_candidate_count,
    verbose: bool = default_verbose,
    wait_seconds: float = default_wait_seconds,
    intent_name: Optional[str] = None,
) -> List[str]:
    # TODO: fix docstring
    """Paraphrase the sentences in an intent using the Gemini Pro model.

    Args:
        original_sentences (Sequence[str]): The original sentences in the intent. Must have at least one sentence.
        vocab (set[str]): The vocabulary.
        before_image (np.ndarray): The before image.
        after_image (np.ndarray): The after image.
        lower_word_count_bound (int, optional): The lower word count bound for the paraphrased sentence.
        upper_word_count_bound (int, optional): The upper word count bound for the paraphrased sentence.
        force_vocab (bool, optional): Whether to force the model to only use words from the vocabulary.
        reattempt_count (int, optional): The number of reattempts to paraphrase the sentence.
        similarity_threshold (float, optional): The similarity threshold for the paraphrased sentences.
        similarity_metric (Literal["meteor", "bleu"], optional): The similarity metric to use for comparing the paraphrased sentences.
        paraphrase_candidate_count (int, optional): The number of paraphrased sentences to generate for each sentence.
        verbose (bool, optional): Whether to print additional information.
        wait_seconds (float, optional): The number of seconds to wait between paraphrasing each sentence.
        intent_name (Optional[str], optional): The name of the intent.

    Returns:
        List[str]: The chosen candidates for the intent.
        The number of chosen candidates is equal to the number of original sentences or
        0 if no candidates could be generated. The paraphrased sentences are chosen based
        on the CLIP score. May contain duplicates. May not contain paraphrased sentences
        for some original sentences.
    """
    # check the constraints
    assert (
        len(original_sentences) > 0
    ), "The intent must have at least one sentence."
    assert (
        similarity_threshold >= 0 and similarity_threshold <= 1
    ), "The similarity threshold must be between 0 and 1."
    assert similarity_metric in {
        "meteor",
        "bleu",
    }, "The similarity metric must be either 'meteor' or 'bleu'."

    # paraphrase each sentence in the intent
    paraphrased_sentences: List[List[str]] = []
    for sentence in original_sentences:
        paraphrased_sentence = paraphrase(
            sentence,
            vocab=vocab,
            lower_word_count_bound=lower_word_count_bound,
            upper_word_count_bound=upper_word_count_bound,
            force_vocab=force_vocab,
            reattempt_count=reattempt_count,
            paraphrase_candidate_count=paraphrase_candidate_count,
        )
        paraphrased_sentences.append(paraphrased_sentence)
        sleep(wait_seconds)

    different_candidates = __remove_similar_sentences(
        list(original_sentences),
        paraphrased_sentences,
        similarity_threshold,
        similarity_metric,
        verbose=verbose,
    )

    # Sanity check to ensure that the number of paraphrased sentences
    # is equal to the number of original sentences
    assert len(different_candidates) == len(original_sentences), (
        f"Something went wrong while paraphrasing {intent_name if intent_name else original_sentences}. "
        "Number of paraphrased sentences does not match the number of original sentences."
    )

    # concatenate the before and after images
    concat_image = convert_opencv_to_pillow_image(
        cv2.hconcat([before_image, after_image])
    )

    # pick the best candidates
    try:
        chosen_candidates = pick_best_candidates(
            different_candidates, concat_image
        )
    except ValueError as e:
        print(f"Error while picking the best candidates: {e}")
        chosen_candidates = []

    if verbose:
        # print the original sentences, different candidates, and chosen candidates
        print(f"Original sentences: {original_sentences}")
        print(f"Different candidates: {different_candidates}")
        print(f"Chosen candidates: {chosen_candidates}")

    return chosen_candidates


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paraphrase a sentence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "karpathy_json", type=str, help="Path to the Karpathy JSON file"
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Path to the image directory (required if --no-clip-score is not specified)",
    )
    parser.add_argument(
        "output_path", type=str, help="Path to the output JSON file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
    )
    parser.add_argument(
        "--no-clip-score",
        "-n",
        action="store_true",
        help="Don't add CLIP score to output.",
    )
    parser.add_argument(
        "--force-vocab",
        "-f",
        action="store_true",
        help="Force the model to only use words from the vocabulary",
    )
    parser.add_argument(
        "--paraphrase-reattempt-count",
        "-r",
        type=int,
        default=default_reattempt_count,
        help="The number of reattempts to paraphrase the sentence, "
        f"if it fails the constraints.",
    )
    parser.add_argument(
        "--lower-word-count-bound",
        "-l",
        type=int,
        default=default_lower_word_count_bound,
        help="The lower word count bound for the paraphrased sentence. ",
    )
    parser.add_argument(
        "--upper-word-count-bound",
        "-u",
        type=int,
        default=default_upper_word_count_bound,
        help="The upper word count bound for the paraphrased sentence. ",
    )
    parser.add_argument(
        "--paraphrase-similarity-threshold",
        "-s",
        type=float,
        default=default_similarity_threshold,
        help="The similarity threshold for the paraphrased sentences. ",
    )
    parser.add_argument(
        "--similarity-metric",
        "-m",
        type=str,
        choices=["meteor", "bleu"],
        default=default_similarity_metric,
        help="The similarity metric to use for comparing the paraphrased sentences. ",
    )
    parser.add_argument(
        "--paraphrase-candidate-count",
        "-c",
        type=int,
        default=default_paraphrase_candidate_count,
        help="The number of paraphrased sentences to generate for each sentence. ",
    )

    args = parser.parse_args(argv)

    return args


def main(
    argv: Optional[Sequence[str]] = None,
) -> int:
    args = get_args()
    # read the Karpathy JSON file
    data = Captions.load(args.karpathy_json)

    # get the vocab
    vocab, eliminated_words = data.get_vocabulary()
    if args.verbose:
        print(f"{vocab=}, {eliminated_words=}")

    # read train and val image pairs with intent
    pairs = data.to_ImagePairs(args.image_dir).pairs

    # randomly select a subset of the pairs
    np.random.seed(0)
    pairs = np.random.choice(
        pairs, 1, replace=False
    )  # for testing # type: ignore

    # paraphrase each sentence in the pairs
    output = dict()
    for pair in tqdm(
        pairs, desc="Paraphrasing pairs", unit="pair", total=len(pairs)
    ):
        if args.verbose:
            print(f"Processing {pair.filename}")

        # save the original sentences
        intent = pair.intent
        original_sentences = intent.get_raw_sentences()
        output[pair.filename] = {
            "original_sentences": original_sentences,
            "paraphrased_sentences": paraphrase_intent(
                original_sentences,
                vocab=vocab,
                lower_word_count_bound=args.lower_word_count_bound,
                upper_word_count_bound=args.upper_word_count_bound,
                force_vocab=args.force_vocab,
                reattempt_count=args.paraphrase_reattempt_count,
                similarity_threshold=args.paraphrase_similarity_threshold,
                similarity_metric=(
                    "meteor" if args.similarity_metric == "meteor" else "bleu"
                ),
                paraphrase_candidate_count=args.paraphrase_candidate_count,
                verbose=args.verbose,
                before_image=pair.imgA.data,
                after_image=pair.imgB.data,
            ),
        }

    if not args.no_clip_score:
        clipScorer = ClipScorer()
        # save the output to a temporary file
        with tempfile.NamedTemporaryFile(
            prefix="paraphrased_sentences_", mode="w", delete=False, dir="."
        ) as f:
            json.dump(output, f, indent=4)

        # get the CLIP score for each sentence
        for pair in tqdm(
            pairs, desc="Getting CLIP scores", unit="pair", total=len(pairs)
        ):
            if args.verbose:
                print(f"Getting CLIP score for {pair.filename}")

            # get the concatenated image
            combined_image = pair.get_concatenated_image()

            # convert the concatenated image to a PIL image
            pil_image = convert_opencv_to_pillow_image(combined_image)

            # get the CLIP score for each sentence
            clip_scores_for_intent = [
                clipScorer.get_clip_score(pil_image, [sentence])
                for sentence in output[pair.filename]["paraphrased_sentences"]
            ]

            output[pair.filename][
                "clip_scores_for_paraphrased_sentences"
            ] = clip_scores_for_intent

        # remove the temporary file
        os.remove(f.name)

    # save the output to a JSON file
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=4)

    return 0


if __name__ == "__main__":
    # TODO: add example usage in the argparse help message
    # TODO: add shorter versions of the arguments to the argparse
    # TODO: make image_dir and --no-clip-score mutually exclusive
    exit(main())
