from __future__ import annotations

import re


def remove_punctuation(string_: str) -> str:
    return re.sub(r"[^\w\s]", "", string_)


def remove_extra_spaces(string_: str) -> str:
    return re.sub(r"\s+", " ", string_).strip()


def normalize_sentence(sentence: str) -> str:
    # strip and lowercase the response text
    lower_cased_and_stripped_sentence = sentence.strip().lower()

    # remove punctuation from the sentence
    sentence_without_punctuation = remove_punctuation(
        lower_cased_and_stripped_sentence,
    )

    # remove any extra spaces
    normalized_sentence = remove_extra_spaces(sentence_without_punctuation)
    return normalized_sentence
