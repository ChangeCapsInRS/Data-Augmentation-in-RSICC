import base64
import json
import os
import re
from typing import List, Literal, Optional

import cv2 as cv
import language_tool_python
import numpy as np
import requests
from PIL import Image
from typing_extensions import deprecated

#######################
#
# Grammar
#
#######################


def get_grammar_checker():
    grammar_checker = language_tool_python.LanguageTool(
        "en-US",
        newSpellings=["bareland", "barelands", "non-vegetated", "highrise"],
        new_spellings_persist=False,
        config={
            "maxCheckTimeMillis": 999999999,
            "requestLimit": 99999999,
        },
    )

    grammar_checker.disabled_rules = {
        "UPPERCASE_SENTENCE_START",  # This rule is disabled because the sentences are supposed to be all lowercase
        "COMMA_COMPOUND_SENTENCE_2",  # This rule is disabled because the sentences does not contain any commas
        "WHITE_HOUSE",  # White House is not a mistake
        "POSSESSIVE_APOSTROPHE",  # The captions does not contain any possessive apostrophes
    }
    # grammar_checker.disable_spellchecking()

    return grammar_checker


grammar_checker = get_grammar_checker()


@deprecated(
    "Can not be used for too many requests. Cannot be used to check >3200 intents at once."
)
def check_sentence_grammar(sentence: str):
    """
    Checks the grammar and spelling of a sentence.

    Args:
        sentence (str): The sentence to check.

    Returns:
        list: The errors in the sentence.
    """
    global grammar_checker

    while True:
        try:
            errors = grammar_checker.check(sentence)
            break
        # after some testing I can say that all hope is lost
        # if the server is not responding...
        # the except blocks below are basically useless
        except (requests.exceptions.ConnectionError, AttributeError):
            grammar_checker.close()
            # grammar_checker = get_grammar_checker()
            grammar_checker._start_server_on_free_port()

    # check/correct bad rules
    for error in errors:
        if error.ruleId == "MOST_SOME_OF_NNS":
            # sort the replacements by length
            # this is to ensure that the shortest replacement is used
            replacements = error.replacements
            sorted_replacements = sorted(replacements, key=lambda x: len(x))
            error.replacements = sorted_replacements

    return errors


@deprecated(
    "Can not be used for too many requests. Cannot be used to correct >3200 intents at once."
)
def correct_sentence_grammar(sentence: str, matches: list):
    """
    Corrects the grammar and spelling of a sentence.

    Args:
        sentence (str): The sentence to correct.
        matches (list): The errors in the sentence.

    Returns:
        str: The corrected sentence.
    """
    return language_tool_python.utils.correct(sentence, matches)


#######################
#
# Spelling
#
#######################

misspelled_words_dict = None
misspelled_words_dict_path = os.path.join("data", "fixed_words.json")

try:
    with open(misspelled_words_dict_path, "r") as f:
        misspelled_words_dict = json.load(f)
except FileNotFoundError:
    print(f"Spelling dict not found at {misspelled_words_dict_path}.")
    # TODO: Maybe use some spell checker library to check the spelling of the words if the dict is not found
    raise NotImplementedError


def check_sentence_spelling(sentence: str) -> set[tuple[str, str]]:
    """
    Checks the spelling of a sentence.

    Args:
        sentence (str): The sentence to check.

    Returns:
        set[tuple[str, str]]: The misspelled words and their correct spelling. [(misspelled_word, correct_spelling)]
    """
    words = sentence.split(" ")
    misspellings = {
        (word, misspelled_words_dict[word])
        for word in words
        if word.lower() in misspelled_words_dict
    }
    return misspellings


def correct_sentence_spelling(
    sentence: str, spelling_errors: Optional[set[tuple[str, str]]] = None
) -> str:
    """
    Corrects the spelling of a sentence.

    Args:
        sentence (str): The sentence to correct.
        spelling_errors (set[tuple[str, str]], optional): The misspelled words and their correct spelling. Defaults to None.

    Returns:
        str: The corrected sentence.
    """

    if spelling_errors is not None:
        spelling_errors_dict = dict(spelling_errors)
    else:
        spelling_errors_dict = misspelled_words_dict

    words = sentence.split(" ")
    corrected_sentence = " ".join(
        (
            spelling_errors_dict[word]
            if word.lower() in spelling_errors_dict
            else word
        )
        for word in words
    )
    return corrected_sentence


def convert_opencv_image_to_base64(image):
    image_bytes = cv.imencode(".jpg", image)[1].tobytes()
    base64_image = base64.b64encode(image_bytes)
    return base64_image.decode("utf-8")


def convert_opencv_to_pillow_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))


def check_alpha_numeric(sentence: str) -> None:
    if re.match("^[a-zA-Z0-9 ]+$", sentence) is None:
        raise ValueError(
            f"Paraphrased sentence contains non-alphanumeric characters: {sentence}"
        )


def check_case(sentence: str) -> None:
    if re.match("^[a-z0-9 ]+$", sentence) is None:
        raise ValueError(
            f"Paraphrased sentence contains uppercase characters: {sentence}"
        )


def check_sentence_length(
    sentence: str, lower_word_count_bound: int, upper_word_count_bound: int
) -> None:
    if len(sentence.split()) < lower_word_count_bound:
        raise ValueError(
            f'Paraphrased sentence is too short ({len(sentence.split())}<{lower_word_count_bound=}): "{sentence}"'
        )
    if len(sentence.split()) > upper_word_count_bound:
        raise ValueError(
            f'Paraphrased sentence is too long ({len(sentence.split())}>{upper_word_count_bound=}): "{sentence}"'
        )


def check_sentence_vocab(sentence: str, vocab: set[str]) -> None:
    # get all the words in the sentence that are not in the vocabulary
    words_not_in_vocab = set(sentence.split()) - vocab

    # check if the sentence contains only words from the vocabulary
    if words_not_in_vocab:
        raise ValueError(
            f"Paraphrased sentence contains words not in the vocabulary: {words_not_in_vocab}"
        )


banned_words = set(
    [
        "paraphrased",
        "sentence",
        "banned",
        "words",
        "word",
        "paraphrase",
        "paraphrasing",
        "paraphraser",
        "paraphrasers",
        "paraphrases",
        "phrase",
        "phrases",
    ]
)


def check_for_banned_words(
    sentence: str,
    banned_words: set[str] = set(
        (
            "paraphrased",
            "sentence",
            "banned",
            "words",
            "word",
            "paraphrase",
            "paraphrasing",
            "paraphraser",
            "paraphrasers",
            "paraphrases",
            "phrase",
            "phrases",
            "final",
        )
    ),
) -> None:
    # get all the words in the sentence that are in the banned words
    banned_words_in_sentence = set(sentence.split()) & banned_words

    # check if the sentence contains any banned words
    if banned_words_in_sentence:
        raise ValueError(
            f"Paraphrased sentence contains banned words: {banned_words_in_sentence}"
        )


def remove_punctuation(string_: str) -> str:
    return re.sub(r"[^\w\s]", "", string_)


def remove_extra_spaces(string_: str) -> str:
    return re.sub(r"\s+", " ", string_).strip()


def normalize_sentence(sentence: str) -> str:
    # strip and lowercase the response text
    lower_cased_and_stripped_sentence = sentence.strip().lower()

    # remove punctuation from the sentence
    sentence_without_punctuation = remove_punctuation(
        lower_cased_and_stripped_sentence
    )

    # remove any extra spaces
    normalized_sentence = remove_extra_spaces(sentence_without_punctuation)
    return normalized_sentence


def get_similar_sentences(
    sentence_to_check: str,
    sentences: List[str],
    similarity_threshold: float,
    similarity_metric: Literal["meteor", "bleu"],
) -> List[tuple[int, str, float]]:
    """Get the sentences that are similar to the sentence to check.

    Args:
        sentence_to_check (str): The sentence to check for similarity.
        sentences (List[str]): The list of sentences to compare with the sentence to check.
        similarity_threshold (float): The similarity threshold for the sentences.
        similarity_metric (Literal["meteor", "bleu"]): The similarity metric to use for comparing the sentences.

    Returns:
        List[tuple[int, str, float]]: A list of tuples containing the index of the similar sentence,
        the similar sentence, and the similarity score.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score

    similarity_scores: list = [
        (
            (
                meteor_score([sentence.split()], sentence_to_check.split())
                if similarity_metric == "meteor"
                else sentence_bleu(
                    [sentence.split()],
                    sentence_to_check.split(),
                    smoothing_function=SmoothingFunction().method4,
                )
            )
            if sentence
            else 0  # if the sentence is empty, set the score to 0
        )
        # check only the previous sentence
        for sentence in sentences
    ]

    # find scores that are greater than the similarity threshold
    similar_sentences = [
        (index, sentence, score)
        for (index, sentence), score in zip(
            enumerate(sentences), similarity_scores
        )
        if score >= similarity_threshold
    ]

    return similar_sentences
