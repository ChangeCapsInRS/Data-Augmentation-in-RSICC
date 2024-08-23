import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

import cv2
import language_tool_python
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StringConstraints,
    ValidationError,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)
from termcolor import cprint
from tqdm import tqdm
from typing_extensions import Self, deprecated

from src.utils import (
    check_sentence_grammar,
    check_sentence_spelling,
    correct_sentence_grammar,
    correct_sentence_spelling,
    normalize_sentence,
)


class Sentence(BaseModel):
    """
    Represents a sentence in the caption JSON file.
    """

    model_config = ConfigDict(
        revalidate_instances="always", validate_assignment=True
    )

    tokens: List[
        Annotated[
            str, StringConstraints(min_length=1, to_lower=True)
        ]  # the tokens are non-empty strings with only words
    ] = Field(
        min_length=2, max_length=50
    )  # the number of tokens is at least 2 and at most 20
    raw: Annotated[
        str,
        StringConstraints(min_length=1, strip_whitespace=True, to_lower=True),
    ]  # the raw is a non-empty string with only words and spaces
    imgid: NonNegativeInt
    sentid: NonNegativeInt

    # @field_validator("raw")
    # @classmethod
    # def remove_extra_spaces_from_raw(cls, v: str) -> str:
    #     """
    #     Removes extra spaces from the raw string.
    #     """
    #     return " ".join(v.split())

    # @field_validator("raw")
    # @classmethod
    # def remove_non_alphanumeric_characters_from_raw(cls, v: str) -> str:
    #     """
    #     Removes non-alphanumeric characters from the raw string.
    #     """
    #     return " ".join([word for word in v.split() if word.isalnum()])

    @field_validator("raw")
    @classmethod
    def normalize_raw(cls, v: str) -> str:
        """
        Normalizes the raw string.
        """
        return normalize_sentence(v)

    @model_validator(mode="before")
    @classmethod
    def set_tokens_as_split_raw(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets the tokens as the split raw string.
        """
        # normalize the raw sentence and set the tokens as the split raw sentence
        raw_sentence = normalize_sentence(v["raw"])
        v["tokens"] = [word for word in raw_sentence.split() if word.isalnum()]
        return v

    # @model_validator(mode="after")
    # def tokens_and_raw_must_have_same_length(self) -> Self:
    #     """
    #     Validates that the number of tokens and the length of the raw string are the same.
    #     """
    #     assert len(self.raw.split()) == len(
    #         self.tokens
    #     ), "The raw and tokens list have different lengths"
    #     return self

    @deprecated(
        (
            "This method cannot be used because the language_tool_python library is not working properly."
            "The server used for grammar checking times out after a certain number of requests (~3200 intents)."
        )
    )
    def check_grammar(self) -> List[language_tool_python.Match]:
        """
        Checks the grammar errors in the sentence.

        Returns:
            List[language_tool_python.Match]: A list of grammar errors in the sentence.
        """
        # Check text for grammar errors
        matches = check_sentence_grammar(self.raw)
        return matches

    def update_with_validation(self, **kwargs):
        """
        ## WARNING: This method may not behave correctly with the checks in the model. Only the given fields are updated.
        Updates the sentence with the given values and validates the model.

        Args:
            **kwargs: The values to update.

        Returns:
            Sentence: The updated sentence.
        """

        updated_self = self.model_copy(update=kwargs)
        validated_self = Sentence.model_validate(updated_self)

        for key in kwargs:
            setattr(self, key, getattr(validated_self, key))

    @deprecated(
        (
            "This method cannot be used because the language_tool_python library is not working properly."
            "The server used for grammar checking times out after a certain number of requests (~3200 intents)."
        )
    )
    def correct_grammar(self, grammar_errors) -> None:
        """
        Corrects the grammar errors in the sentence.
        """
        # Correct grammar errors
        corrected_raw_sentence = correct_sentence_grammar(
            self.raw, grammar_errors
        )
        try:
            corrected_token = corrected_raw_sentence.split()
            self.update_with_validation(
                raw=corrected_raw_sentence, tokens=corrected_token
            )
        except ValidationError as e:
            print(
                f"Encountered an error while correcting the grammar of the sentence: {self.raw=} {corrected_raw_sentence=} {grammar_errors=} {e}"
            )

    def check_spelling(self):
        """
        Checks the spelling of the sentence.

        Returns:
            set[tuple[str, str]]: The misspelled words and their correct spelling. [(misspelled_word, correct_spelling)]
        """
        return check_sentence_spelling(self.raw)

    def correct_spelling(self, spelling_errors):
        """
        Corrects the spelling errors in the sentence.
        """
        corrected_raw_sentence = correct_sentence_spelling(
            self.raw, spelling_errors
        )
        corrected_token = corrected_raw_sentence.split()
        try:
            self.update_with_validation(
                raw=corrected_raw_sentence, tokens=corrected_token
            )
        except ValidationError as e:
            print(
                f"Encountered an error while correcting the spelling of the sentence: {self.raw=} {corrected_raw_sentence=} {spelling_errors=} {e}"
            )


class Intent(BaseModel):
    """
    Represents an intent in the caption JSON file.
    """

    model_config = ConfigDict(
        revalidate_instances="always", validate_assignment=True
    )

    sentences: List[Sentence] = Field(
        min_length=1
    )  # the number of sentences is at least 1
    changeflag: Annotated[
        int, Field(ge=0, le=1, strict=True)
    ]  # the changeflag is 0 or 1
    filepath: Literal["train", "val", "test"]
    split: Literal["train", "val", "test"]
    imgid: NonNegativeInt = Field(strict=True)
    sentids: List[NonNegativeInt] = Field(
        min_length=1
    )  # the number of sentids is at least 1
    filename: Annotated[
        str, StringConstraints(min_length=1, strict=True)
    ]  # the filename is a non-empty string

    # ? It may be better to not change the number of sentences
    # @field_validator("sentences")
    # @classmethod
    # def sentences_must_have_five_items(cls, v):
    #     """
    #     Validates that the sentences list has 5 items.
    #     """
    #     assert len(v) != 0, "The sentences list is empty"

    #     if len(v) == 5:
    #         return v

    #     if len(v) > 5:
    #         return v[:5]

    #     # if the number of sentences is less than 5, repeat the sentences
    #     for i in range(len(v), 5):
    #         new_sentence = v[0].copy()
    #         new_sentence.sentid = i
    #         v.append(new_sentence)

    #     # sanity check
    #     assert len(v) == 5, "The number of sentences is not 5"
    #     return v

    @field_validator("sentences")
    @classmethod
    def sentences_must_have_consecutive_sentid(cls, v):
        """
        Validates that the sentids of the sentences are consecutive.
        """
        sentids = [sentence.sentid for sentence in v]
        assert sentids == list(
            range(min(sentids), max(sentids) + 1)
        ), "The sentids of the sentences must be consecutive"
        return v

    @field_validator("sentences")
    @classmethod
    def all_sentences_must_have_same_imgid(cls, v):
        """
        Validates that the imgid of the sentences is the same.
        """
        imgids = list(set(sentence.imgid for sentence in v))
        assert len(imgids) == 1, "The imgid of the sentences must be the same"
        return v

    @model_validator(mode="after")
    def sentences_must_have_same_imgid_as_intent(self) -> Self:
        """
        Validates that the imgid of the sentences is the same as the intent.
        """
        assert all(
            sentence.imgid == self.imgid for sentence in self.sentences
        ), "The imgid of the sentences must be the same as the intent"
        return self

    @model_validator(mode="after")
    def sentids_must_be_same_as_sentences(self) -> Self:
        # ? This assumes the sentids are in the same order as the sentences
        """
        Validates that the sentids of the sentences are the same as the intent.
        """
        assert all(
            sentence.sentid == self.sentids[i]
            for i, sentence in enumerate(self.sentences)
        ), "The sentids of the sentences must be the same as the intent"
        return self

    # ? This assumes the filepath is the same as the split (which is what we see in the data)
    @model_validator(mode="after")
    def filepath_must_be_same_as_split(self) -> Self:
        """
        Validates that the filepath is the same as the split.
        """
        assert (
            self.filepath == self.split
        ), "The filepath must be the same as the split"
        return self

    def check_for_errors(self) -> List[Dict[str, Any]]:
        """
        Checks for spelling and grammar errors in the sentences of the intent.

        Returns:
            List[Dict[str, Any]]: A list of errors in the sentences of the intent.
        """
        errors = []
        for i, sentence in enumerate(self.sentences):
            errors_in_sentence = dict()
            # grammar_errors = sentence.check_grammar()
            # if grammar_errors:
            #     errors_in_sentence["grammar_errors"] = grammar_errors

            spelling_errors = sentence.check_spelling()
            if spelling_errors:
                errors_in_sentence["spelling_errors"] = spelling_errors

            # if there are errors in the sentence
            # save sentence index and errors to the list
            if errors_in_sentence:
                errors_in_sentence["sentence_index"] = i
                errors.append(errors_in_sentence)
        return errors

    def fix_errors(self, errors: Optional[List[Dict]] = None) -> Self:
        """
        Fixes the spelling and grammar errors in the sentences of the intent.

        Args:
            errors (Optional[List[dict]]): The errors in the sentences of the intent.

        Returns:
            Self: The intent with the fixed errors.
        """
        if not errors:
            errors = self.check_for_errors()

        for error in errors:
            # get the sentence with the error
            sentence: Sentence = self.sentences[error["sentence_index"]]
            # if "grammar_errors" in error:
            #     sentence.correct_grammar(error["grammar_errors"])

            if "spelling_errors" in error:
                sentence.correct_spelling(error["spelling_errors"])
            # ? Can we remove this line?
            self.sentences[error["sentence_index"]] = sentence

        return self

    def get_raw_sentences(self) -> List[str]:
        """
        Returns the raw sentences of the intent.
        """
        return [sentence.raw for sentence in self.sentences]


class Captions(BaseModel):
    """
    Represents the captions JSON file.
    """

    model_config = ConfigDict(
        revalidate_instances="always", validate_assignment=True
    )

    images: List[Intent] = Field(
        min_length=1
    )  # the number of intents is at least 1

    @field_validator("images")
    @classmethod
    def filenames_must_be_unique(cls, v):
        """
        Validates that the filename of the intents is unique.
        """
        filenames = [intent.filename for intent in v]
        duplicate_filenames = {
            filename for filename in filenames if filenames.count(filename) > 1
        }
        assert (
            not duplicate_filenames
        ), f"The following {len(duplicate_filenames)} filename(s) are duplicated: {', '.join(duplicate_filenames)}"
        return v

    @field_validator("images")
    @classmethod
    def imgids_must_be_unique(cls, v):
        """
        Validates that the imgid of the intents is unique.
        """
        imgids = [intent.imgid for intent in v]
        assert len(imgids) == len(
            set(imgids)
        ), "The imgid of the intents must be unique"
        return v

    def add_intent(self, intent: Intent) -> None:
        """
        Adds an intent to the list of intents.

        Args:
            intent (Intent): The intent to add.
        """
        self.images.append(intent)

        self = Captions.model_validate(self)

    def to_ImagePairs(
        self,
        images_path: str,
        splits: set[str] = {"train", "val"},
        levircc=False,
    ) -> "ImagePairs":
        """
        Converts the captions to a list of pairs of images with intents.

        Args:
            images_path (str): The path of the images.
            splits (set[str]): Image pair from intent is taken only if its 'split' field\
            is one of the strings in 'splits'. Allowed values are 'train', 'val', 'test'.\
            Defaults to {"train", "val"}.

        Returns:
            List[ImagePairsWithIntent]: A list of pairs of images with intents.
        """
        # validate the splits
        assert all(
            split in {"train", "val", "test"} for split in splits
        ), "The splits must be 'train', 'val', or 'test'"

        # print("levircc", levircc)
        results = []
        for intent in self.images:
            if intent.split in splits:
                imgA, imgB = Image.load_pair_of_images(
                    images_path, intent.filename, intent.split, levircc
                )
                results.append(
                    ImagePairWithIntent(imgA=imgA, imgB=imgB, intent=intent)
                )

        return ImagePairs(pairs=results)

    @staticmethod
    def check_errors_in_sentences_(
        intents: List[Intent],
    ) -> List[Dict[str, Any]]:
        """
        Checks for spelling and grammar errors in the sentences of the intents.

        Args:
            intents (List[Intent]): The list of intents.

        Returns:
            List[Dict[str, Any]]: A list of errors in the sentences of the intents.
        """
        check_errors = lambda x: {"intent": x, "errors": x.check_for_errors()}

        with ThreadPoolExecutor() as executor:
            error_checks = list(
                tqdm(
                    executor.map(check_errors, intents),
                    total=len(intents),
                    desc="Checking errors",
                    unit="intent",
                )
            )

        errors = [error for error in error_checks if error["errors"]]

        return errors

    def check_errors_in_sentences(self) -> List[Dict[str, Any]]:
        """
        Checks for spelling and grammar errors in the sentences of the intents.

        Returns:
            List[Dict[str, Any]]: A list of errors in the sentences of the intents.
        """

        return Captions.check_errors_in_sentences_(self.images)

    def fix_errors_in_sentences(
        self,
        errors: Optional[List[Dict[str, Any]]] = None,
        max_iterations: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fixes the spelling and grammar errors in the sentences of the intents.

        Args:
            errors (Optional[List[Dict[str, Any]]]): The errors \
                in the sentences of the intents. Defaults to None.
            max_iterations (Optional[int]): The maximum number of \
                iterations to fix the errors. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: The list of remaining errors in the sentences of the intents.
        """
        if not errors:
            errors = self.check_errors_in_sentences()
        if not max_iterations:
            max_iterations = 5

        if len(errors) == 0:
            print("No errors found")
            return []

        # TODO: For some reason the grammar checker server times out after ~3200 requests, so we need to fix the server or find a workaround
        # Fix the errors until there are no errors or the maximum number of iterations is reached
        current_iteration = 1
        while errors and current_iteration <= max_iterations:
            with ThreadPoolExecutor() as executor:
                fixed_intents = list(
                    tqdm(
                        executor.map(
                            lambda x: x["intent"].fix_errors(x["errors"]),
                            errors,
                        ),
                        total=len(errors),
                        desc=f"Fixing errors (iteration {current_iteration}/{max_iterations})",
                        unit="intent",
                    )
                )

            current_iteration += 1

            # check for errors again
            errors = Captions.check_errors_in_sentences_(fixed_intents)

        if errors:
            # print the remaining errors
            print(
                f"WARNING: The following {len(errors)} intent(s) still have errors:"
            )
            for error in errors:
                print(error["intent"].filename)
        else:
            print("All errors have been fixed")

        # ? This might be overkill, considering the Intent.fix_errors \
        # method already validates the model
        return errors

    def get_vocabulary(
        self, minimum_occurrences: int = 5
    ) -> tuple[set[str], set[str]]:
        """
        Gets the vocabulary from the captions.

        Args:
            minimum_occurrences (int): The minimum number of occurrences \
                of a word to be included in the vocabulary. Defaults to 5.

        Returns:
            tuple[set[str], set[str]]: A tuple containing the vocabulary and the eliminated words.
        """
        vocabulary = set()
        eliminated_words = set()
        counter = Counter()
        for intent in self.images:
            for sentence in intent.sentences:
                counter.update(sentence.tokens)

        for word, count in counter.items():
            if count >= minimum_occurrences:
                vocabulary.add(word)
            else:
                eliminated_words.add(word)

        return vocabulary, eliminated_words

    @staticmethod
    def normalize_dict(captions_dict: Dict[str, Any]) -> "Captions":
        """
        Normalizes the captions.

        Args:
            captions_dict (Dict[str, Any]): The captions.

        Returns:
            Captions: The normalized captions.
        """

        # sort the captions by filename
        captions_dict["images"] = sorted(
            captions_dict["images"], key=lambda x: x["filename"]
        )

        # set the imgid and sentid to 0
        imgid = 0
        sentid = 0

        for intent in captions_dict["images"]:
            # set the imgid of the intent
            intent["imgid"] = imgid

            # sort the sentences by raw
            # intent["sentences"] = sorted(intent["sentences"], key=lambda x: x["raw"])

            # set the sentid and imgid of the sentences
            for sentence in intent["sentences"]:
                sentence["imgid"] = imgid
                sentence["sentid"] = sentid
                sentid += 1

                # normalize raw sentence and update tokens
                # sentence["raw"] = normalize_sentence(sentence["raw"])
                # sentence["tokens"] = sentence["raw"].split()

            # set the sentids of the intent
            intent["sentids"] = [
                sentence["sentid"] for sentence in intent["sentences"]
            ]
            imgid += 1

        return Captions.model_validate(captions_dict)

    @staticmethod
    def load(json_path: str, *, fixes_errors: bool = False) -> "Captions":
        """
        Loads the captions from the given JSON file.

        Args:
            json_path (str): The path of the JSON file.
            fixes_errors (bool): Whether to fix the errors in the sentences. Defaults to False.

        Returns:
            Captions: The captions loaded from the JSON file.
        """

        with open(json_path, "r") as f:
            captions = Captions.model_validate_json(f.read())

        if fixes_errors:
            errors: List[Dict[str, Any]] = captions.check_errors_in_sentences()
            if len(errors) > 0:
                cprint("WARNING: input file has errors", "red")
                captions.fix_errors_in_sentences(errors)
        return captions


class Image(BaseModel):
    """
    Represents an image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: cv2.typing.MatLike
    filename: Annotated[
        str, StringConstraints(min_length=1)
    ]  # the fileName is a non-empty string

    @staticmethod
    def load_pair_of_images(
        base_path: str,
        filename: str,
        split: Literal["test", "train", "val"],
        levircc=False,
    ) -> tuple["Image", "Image"]:
        """
        Reads a pair of images from the given base path.

        Args:
            base_path (str): The base path of the images.
            filename (str): The file name of the images.
            split (Literal["test", "train", "val"]): The split of the images.

        Returns:
            tuple[Image, Image]: A tuple containing the pair of images.
        """
        # print("levircc", levircc)
        imageA_path, imageB_path = Image.get_paths_for_pair_of_images(
            base_path, split, filename, levircc
        )
        imgA = cv2.imread(imageA_path)
        imgB = cv2.imread(imageB_path)

        return Image(data=imgA, filename=filename), Image(
            data=imgB, filename=filename
        )

    @staticmethod
    def write_pair_of_images(
        base_path: str,
        split: Literal["test", "train", "val"],
        imageA: "Image",
        imageB: "Image",
        *,
        overwrite: bool = False,
        make_dirs: bool = True,
        dataset: Literal["LevirCC", "SecondCC"] = "SecondCC",
    ):
        """
        Writes a pair of images to the given base path.

        Args:
            base_path (str): The base path of the images.
            split (Literal["test", "train", "val"]): The split of the images.
            imageA (Image): The first image.
            imageB (Image): The second image.
            overwrite (bool): Whether to overwrite the existing images. Defaults to False.
            make_dirs (bool): Whether to create the directories if they do not exist.\
                Defaults to True.
        """
        imageA_path, imageB_path = Image.get_paths_for_pair_of_images(
            base_path, split, imageA.filename, levircc=(dataset == "LevirCC")
        )

        if make_dirs:
            os.makedirs(os.path.dirname(imageA_path), exist_ok=True)
            os.makedirs(os.path.dirname(imageB_path), exist_ok=True)

        if not overwrite:
            assert not os.path.exists(
                imageA_path
            ), f"File {imageA_path} already exists"
            assert not os.path.exists(
                imageB_path
            ), f"File {imageB_path} already exists"

        cv2.imwrite(imageA_path, imageA.data)
        cv2.imwrite(imageB_path, imageB.data)

    @staticmethod
    def get_paths_for_pair_of_images(
        base_path: str,
        split: Literal["test", "train", "val"],
        filename: str,
        levircc=False,
    ) -> tuple[str, str]:
        """
        Returns the path of the image.

        Args:
            base_path (str): The base path of the images.
            split (Literal["test", "train", "val"]): The split of the images.
            filename (str): The file name of the image.

        Returns:
            tuple[str, str]: A tuple containing the paths of the images.
        """
        # print("levircc", levircc)
        if not levircc:
            return (
                os.path.join(base_path, split, "A", f"{split}_{filename}"),
                os.path.join(base_path, split, "B", f"{split}_{filename}"),
            )
        else:
            return (
                os.path.join(base_path, split, "A", f"{filename}"),
                os.path.join(base_path, split, "B", f"{filename}"),
            )


class ImagePairWithIntent(BaseModel):
    """
    Represents a pair of images with an intent.
    """

    imgA: Image
    imgB: Image
    intent: Intent

    @model_validator(mode="after")
    def images_and_intent_must_have_same_filename(self) -> Self:
        """
        Validates that the imgA, imgB, and intent fileName are the same.
        """
        assert (
            self.imgA.filename == self.imgB.filename == self.intent.filename
        ), "The imgA, imgB, and intent filename must be the same"
        return self

    @computed_field
    @property
    def filename(self) -> str:
        """
        Returns the file name of the image pair.
        """
        return self.imgA.filename

    # setter for filename
    @filename.setter
    def filename(self, value: str) -> None:
        """
        Sets the file name of the image pair.
        """
        self.imgA.filename = value
        self.imgB.filename = value
        self.intent.filename = value

    @computed_field
    @property
    def split(self) -> Literal["test", "train", "val"]:
        """
        Returns the split of the intent.
        """
        return self.intent.split

    # This method is called when the object is serialized into a dictionary
    @model_serializer()
    def serialize(self) -> Dict[str, Any]:
        """
        Serializes the image pair and the intent to a dictionary.
        """
        return {
            self.filename: {
                "before": self.imgA.data,
                "after": self.imgB.data,
                "sentences": [
                    sentence.model_dump() for sentence in self.intent.sentences
                ],
            }
        }

    def get_concatenated_image(self) -> Any:
        """
        Concatenates the before and after images horizontally.

        Returns:
            Any: The concatenated image.
        """
        return cv2.hconcat([self.imgA.data, self.imgB.data])

    def st_write(self) -> None:
        """
        Writes the image pair and the intent to the Streamlit app.
        """
        import streamlit as st

        before_image_column, after_image_column, intent_column = st.columns(
            [1, 1, 2]
        )

        with before_image_column:
            st.image(
                self.imgA.data, use_column_width=True, caption="Before Image"
            )

        with after_image_column:
            st.image(
                self.imgA.data, use_column_width=True, caption="After Image"
            )

        with intent_column:
            st.write("**Captions**")
            with st.container(border=True):
                st.text(
                    "\n".join(
                        [sentence.raw for sentence in self.intent.sentences]
                    )
                )

        # st.divider()
        # st.write(f"File Name: `{self.fileName}` Category: `{self.intent.category}`")

    def augment(
        self, augment_function: Callable, **kwargs
    ) -> "ImagePairWithIntent":
        """
        Augments the image pair with the given augmentation function.

        Args:
            augment_function (Callable): The augmentation function.
            **kwargs: The keyword arguments for the augmentation function.

        Returns:
            ImagePairWithIntent: The augmented image pair.
        """
        (
            augmented_before_image,
            augmented_after_image,
            augmented_raw_sentences,
        ) = augment_function(
            before_image=self.imgA.data,
            after_image=self.imgB.data,
            raw_sentences=self.intent.get_raw_sentences(),
            **kwargs,
        )

        # sanity check
        assert len(augmented_raw_sentences) == len(self.intent.sentences)

        # the new sentences for the augmented image pair
        augmented_sentences = [
            Sentence(
                raw=raw_sentence,
                tokens=raw_sentence.split(),
                imgid=self.intent.imgid,
                sentid=self.intent.sentids[i],
            ).model_dump(exclude_unset=True, exclude_defaults=True)
            for i, raw_sentence in enumerate(augmented_raw_sentences)
        ]

        # the new values for the augmented image pair
        new_values = {
            **self.intent.model_dump(
                exclude_unset=True, exclude_defaults=True
            ),  # copy all *SET* fields from the original intent
            "sentences": augmented_sentences,  # replace the sentences field with the new sentences
        }

        # get the set fields of the model
        set_fields = self.intent.model_fields_set
        set_fields.add(
            "sentences"
        )  # add the sentences field to the set fields (just in case it's not there)

        # create a new intent with the new values
        new_intent = Intent.model_construct(
            set_fields=set_fields,
            **new_values,
        )

        return ImagePairWithIntent(
            imgA=Image(
                data=augmented_before_image, filename=self.imgA.filename
            ),
            imgB=Image(
                data=augmented_after_image, filename=self.imgB.filename
            ),
            intent=new_intent,
        )


class ImagePairs(BaseModel):
    pairs: List["ImagePairWithIntent"]

    @staticmethod
    def load(
        json_path: str,
        images_path: str,
        fixes_errors: bool = False,
        splits: set[str] = {"train", "val"},
    ) -> "ImagePairs":
        """
        Loads a list of pairs of images with intents from the given JSON file and images.

        Args:
            json_path (str): The path of the JSON file.
            images_path (str): The path of the images.
            fixes_errors (bool): Whether to fix the errors in the sentences. Defaults to False.
            splits (set[str]): Image pair from intent is taken only if its 'split' field

        Returns:
            List[ImagePairsWithIntent]: A list of pairs of images with intents.
        """
        captions = Captions.load(json_path, fixes_errors=fixes_errors)
        return captions.to_ImagePairs(images_path, splits)

    def augment(
        self,
        augment_function: Callable,
        *,
        augmenter_name: Optional[str] = None,
        **kwargs,
    ) -> "AugmentedImagePairs":
        """
        Augments the image pairs with the given augmentation function.

        Args:
            augment_function (Callable): The augmentation function.
            augmenter_name (Optional[str]): The name of the augmentation.
            If not provided, the name of the augmentation function is used. Defaults to None.
            **kwargs: The keyword arguments for the augmentation function.

        Returns:
            AugmentedImagePairs: The augmented image pairs.
        """
        augmenter_name = augmenter_name or augment_function.__name__
        new_pairs = [
            pair.augment(augment_function, **kwargs)
            for pair in tqdm(
                self.pairs,
                desc=augmenter_name,
                unit="intent",
                total=len(self.pairs),
            )
        ]
        return AugmentedImagePairs(
            pairs=new_pairs,
            augmenter_name=augmenter_name,
        )

    def random_augment(
        self,
        augment_function: Callable,
        *,
        probability: float = 0.5,
        augmenter_name: Optional[str] = None,
        **kwargs,
    ) -> "AugmentedImagePairs":
        """
        Augments the image pairs with the given augmentation function with a random probability.

        Args:
            augment_function (Callable): The augmentation function.
            probability (float): The probability of applying the augmentation function. Defaults to 0.5.
            augmenter_name (Optional[str]): The name of the augmentation.
            If not provided, the name of the augmentation function is used. Defaults to None.
            **kwargs: The keyword arguments for the augmentation function.

        Returns:
            AugmentedImagePairs: The augmented image pairs.
        """
        augmenter_name = augmenter_name or augment_function.__name__
        new_pairs = [
            (
                pair.augment(augment_function, **kwargs)
                if random.random() < probability
                else pair
            )
            for pair in tqdm(
                self.pairs,
                desc=augmenter_name,
                unit="intent",
                total=len(self.pairs),
            )
        ]
        return AugmentedImagePairs(
            pairs=new_pairs, augmenter_name=augmenter_name
        )


class AugmentedImagePairs(ImagePairs):
    """
    Represents a set of augmented image pairs.
    """

    # a non-empty string
    augmenter_name: Annotated[str, StringConstraints(min_length=1)]

    def save(
        self,
        output_path: str,
        *,
        overwrite: bool = False,
        make_dirs: bool = True,
        dataset: Literal["LevirCC", "SecondCC"] = "SecondCC",
    ):
        """
        Saves the augmented image pairs to the given output path.

        The augmented image pairs are saved in the following tree structure:

        ```
        {output_path}
        ├── train
        │   ├── A
        │   └── B
        ├── val
        │   ├── A
        │   └── B
        ├── test
        │   ├── A
        │   └── B
        └── {self.augmenter_name}.json
        ```


        Args:
            output_path (str): The output path.
        """
        # add the augmentation name to the filenames
        for pair in self.pairs:
            pair.filename = self.__get_augmented_filename(
                output_path, pair, overwrite
            )

        # normalize the augmented data
        captions = Captions.normalize_dict(
            {"images": [pair.intent.model_dump() for pair in self.pairs]}
        )

        # get the path of the json file
        json_path = os.path.join(
            output_path, self.__get_json_filename(output_path, overwrite)
        )

        if make_dirs:
            # create directories if they don't exist
            os.makedirs(os.path.dirname(json_path), exist_ok=True)

        # write the augmented data to a json file
        with open(json_path, "x" if not overwrite else "w") as f:
            f.write(captions.model_dump_json(indent=4))

        # write images to folders
        for pair in self.pairs:
            Image.write_pair_of_images(
                output_path,
                pair.split,
                pair.imgA,
                pair.imgB,
                dataset=dataset,
                overwrite=overwrite,
                make_dirs=make_dirs,
            )
        print(
            f"{len(self.pairs)} {self.augmenter_name!r} augmented image pairs saved to {output_path}"
        )

    def __get_augmented_filename(
        self,
        output_path: str,
        pair: ImagePairWithIntent,
        overwrite: bool = False,
    ) -> str:
        old_filename = pair.filename
        old_filename_root, old_filename_extension = os.path.splitext(
            old_filename
        )

        # create a new filename with the augmenter name
        new_filename = f"{old_filename_root}_{self.augmenter_name}{old_filename_extension}"

        if overwrite:
            return new_filename

        counter = 1
        # check if the new filename already exists
        while any(
            os.path.exists(image_path)
            for image_path in Image.get_paths_for_pair_of_images(
                output_path, pair.split, new_filename
            )
        ):
            # if it does, add a counter to the filename
            new_filename = f"{old_filename_root}_{self.augmenter_name}_{counter}{old_filename_extension}"
            counter += 1

        return new_filename

    def __get_json_filename(
        self, output_path: str, overwrite: bool = False
    ) -> str:
        json_filename = f"{self.augmenter_name}.json"

        if overwrite:
            return json_filename

        counter = 1
        # check if the new filename already exists
        while os.path.exists(os.path.join(output_path, json_filename)):
            # if it does, add a counter to the filename
            json_filename = f"{self.augmenter_name}_{counter}.json"
            counter += 1

        return json_filename
