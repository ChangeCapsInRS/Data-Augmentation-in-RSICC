import copy
from typing import Callable, List, Literal, Sequence, Tuple

import cv2 as cv
import numpy as np

from src.random_augment import RandomAugmenter

BOTTOM_DIRECTIONS = (
    "bottom",
    "lower",
    "below",
    "bottommost",
    "lowest",
)

TOP_DIRECTIONS = (
    "top",
    "upper",
    "above",
    "topmost",
    "highest",
    "higher",
)


def vertical_mirror(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        horizontal_mirrored_before_image = np.flip(before_image, axis=0)
        horizontal_mirrored_after_image = np.flip(after_image, axis=0)
        return (
            horizontal_mirrored_before_image,
            horizontal_mirrored_after_image,
        )

    def augment_text():
        top_directions = TOP_DIRECTIONS
        bottom_directions = BOTTOM_DIRECTIONS
        new_sentences = []
        for sentence in raw_sentences:
            words = sentence.split(" ")
            for k in range(len(words)):
                if words[k] in top_directions:
                    words[k] = bottom_directions[0]
                elif words[k] in bottom_directions:
                    words[k] = top_directions[0]
            new_sentences.append(" ".join(words))

        return new_sentences

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def horizontal_mirror(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        vertical_mirrored_before_image = np.flip(before_image, axis=1)
        vertical_mirrored_after_image = np.flip(after_image, axis=1)
        return vertical_mirrored_before_image, vertical_mirrored_after_image

    def augment_text():
        left_direction = "left"
        right_direction = "right"
        new_sentences = []
        for sent in raw_sentences:
            words = sent.split(" ")
            for k in range(len(words)):
                if words[k] == left_direction:
                    words[k] = right_direction
                elif words[k] == right_direction:
                    words[k] = left_direction
            new_sentences.append(" ".join(words))

        return new_sentences

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def left_diagonal_mirror(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        left_diagonal_mirrored_before_image = cv.transpose(before_image)
        left_diagonal_mirrored_after_image = cv.transpose(after_image)
        return (
            left_diagonal_mirrored_before_image,
            left_diagonal_mirrored_after_image,
        )

    def augment_text():
        right_direction = "right"
        left_direction = "left"
        bottom_directions = BOTTOM_DIRECTIONS
        top_directions = TOP_DIRECTIONS
        new_sentences = []
        for sent in raw_sentences:
            words = sent.split(" ")
            k = 0
            while k < len(words):
                if (
                    k + 1 < len(words)
                    and words[k] in bottom_directions
                    and words[k + 1] == left_direction
                ):
                    words[k], words[k + 1] = top_directions[0], right_direction
                    k += 1
                elif (
                    k + 1 < len(words)
                    and words[k] in top_directions
                    and words[k + 1] == right_direction
                ):
                    words[k], words[k + 1] = (
                        bottom_directions[0],
                        left_direction,
                    )
                    k += 1
                elif (
                    k + 1 < len(words)
                    and words[k] in top_directions
                    and words[k + 1] == left_direction
                ):
                    k += 1
                elif (
                    k + 1 < len(words)
                    and words[k] in bottom_directions
                    and words[k + 1] == right_direction
                ):
                    k += 1
                elif words[k] == right_direction:
                    words[k] = bottom_directions[0]
                elif words[k] in bottom_directions:
                    words[k] = right_direction
                elif words[k] == left_direction:
                    words[k] = top_directions[0]
                elif words[k] in top_directions:
                    words[k] = left_direction
                k += 1
            new_sentences.append(" ".join(words))

        return new_sentences

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def blur(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        kernel_size = 5
        blurred_before_image = cv.GaussianBlur(
            before_image, (kernel_size, kernel_size), 0
        )
        blurred_after_image = cv.GaussianBlur(
            after_image, (kernel_size, kernel_size), 0
        )

        return blurred_before_image, blurred_after_image

    def augment_text():
        return list(copy.deepcopy(raw_sentences))

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def augment_text_rotate_90(raw_sentences: Sequence[str]):
    right_direction = "right"
    left_direction = "left"
    bottom_directions = BOTTOM_DIRECTIONS
    top_directions = TOP_DIRECTIONS
    new_sentences = []
    for sent in raw_sentences:
        words = sent.split(" ")
        k = 0
        while k < len(words):
            if (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = top_directions[0], left_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], right_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = top_directions[0], right_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], left_direction
                k += 1
            elif words[k] == right_direction:
                words[k] = bottom_directions[0]
            elif words[k] in bottom_directions:
                words[k] = left_direction
            elif words[k] == left_direction:
                words[k] = top_directions[0]
            elif words[k] in top_directions:
                words[k] = right_direction
            k += 1
        new_sentences.append(" ".join(words))

    return new_sentences


def augment_text_rotate_270(raw_sentences: Sequence[str]):
    right_direction = "right"
    left_direction = "left"
    bottom_directions = BOTTOM_DIRECTIONS
    top_directions = TOP_DIRECTIONS
    new_sentences = []
    for sent in raw_sentences:
        words = sent.split(" ")
        k = 0
        while k < len(words):
            if (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], right_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = top_directions[0], left_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], left_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = top_directions[0], right_direction
                k += 1
            elif words[k] == right_direction:
                words[k] = top_directions[0]
            elif words[k] in bottom_directions:
                words[k] = right_direction
            elif words[k] == left_direction:
                words[k] = bottom_directions[0]
            elif words[k] in top_directions:
                words[k] = left_direction
            k += 1
        new_sentences.append(" ".join(words))

    return new_sentences


def augment_text_rotate_180(raw_sentences: Sequence[str]):
    right_direction = "right"
    left_direction = "left"
    bottom_directions = BOTTOM_DIRECTIONS
    top_directions = TOP_DIRECTIONS
    new_sentences = []
    for sent in raw_sentences:
        words = sent.split(" ")
        k = 0
        while k < len(words):
            if (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = top_directions[0], right_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], left_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in top_directions
                and words[k + 1] == left_direction
            ):
                words[k], words[k + 1] = bottom_directions[0], right_direction
                k += 1
            elif (
                k + 1 < len(words)
                and words[k] in bottom_directions
                and words[k + 1] == right_direction
            ):
                words[k], words[k + 1] = top_directions[0], left_direction
                k += 1
            elif words[k] == right_direction:
                words[k] = left_direction
            elif words[k] in bottom_directions:
                words[k] = top_directions[0]
            elif words[k] == left_direction:
                words[k] = right_direction
            elif words[k] in top_directions:
                words[k] = bottom_directions[0]
            k += 1
        new_sentences.append(" ".join(words))

    return new_sentences


def rotate(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    *,
    degree: Literal[90, 180, 270],
):
    assert degree in [90, 180, 270], "Degree must be 90, 180, or 270"

    def augment_images():
        degree_to_opencv_values = {
            90: cv.ROTATE_90_CLOCKWISE,
            180: cv.ROTATE_180,
            270: cv.ROTATE_90_COUNTERCLOCKWISE,
        }
        rotated_before_image = cv.rotate(
            before_image, degree_to_opencv_values[degree]
        )
        rotated_after_image = cv.rotate(
            after_image, degree_to_opencv_values[degree]
        )
        return rotated_before_image, rotated_after_image

    def augment_text():
        if degree == 90:
            return augment_text_rotate_90(raw_sentences)
        elif degree == 180:
            return augment_text_rotate_180(raw_sentences)
        elif degree == 270:
            return augment_text_rotate_270(raw_sentences)

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def rotate_90(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    return rotate(before_image, after_image, raw_sentences, degree=90)


def rotate_180(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    return rotate(before_image, after_image, raw_sentences, degree=180)


def rotate_270(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    return rotate(before_image, after_image, raw_sentences, degree=270)


def brighten_both(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    *,
    alpha=1.0,
    beta=80,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        # new_pixel = ALPHA * old_pixel + BETA

        before_augmented = cv.convertScaleAbs(
            before_image, alpha=alpha, beta=beta
        )
        after_augmented = cv.convertScaleAbs(
            after_image, alpha=alpha, beta=beta
        )

        return before_augmented, after_augmented

    def augment_text():
        return list(copy.deepcopy(raw_sentences))

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def brighten_before(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    *,
    alpha: float = 1.0,
    beta: float = 80.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        # new_pixel = ALPHA * old_pixel + BETA

        before_augmented = cv.convertScaleAbs(
            before_image, alpha=alpha, beta=beta
        )

        return before_augmented, after_image

    def augment_text():
        return list(copy.deepcopy(raw_sentences))

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def brighten_after(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    *,
    alpha: float = 1.0,
    beta: float = 80.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    def augment_images():
        # new_pixel = ALPHA * old_pixel + BETA

        after_augmented = cv.convertScaleAbs(
            after_image, alpha=alpha, beta=beta
        )

        return before_image, after_augmented

    def augment_text():
        return list(copy.deepcopy(raw_sentences))

    augmented_images = augment_images()
    return augmented_images[0], augmented_images[1], augment_text()


def random_augment(
    before_image: np.ndarray,
    after_image: np.ndarray,
    raw_sentences: Sequence[str],
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    version = 1

    if version == 1:
        augmenters = [
            [[horizontal_mirror]],
            [[vertical_mirror]],
            [[rotate_90]],
            [[brighten_both]],
        ]
    elif version == 2:
        augmenters = [
            [
                [horizontal_mirror, vertical_mirror, left_diagonal_mirror],
                [rotate_90, rotate_180, rotate_270],
            ],
            [[brighten_both, brighten_before, brighten_after], [blur]],
        ]
    elif version == 3:
        augmenters = [
            [[rotate_90, rotate_180, rotate_270]],
            [[blur]],
        ]
    elif version == 4:
        augmenters = [
            [[rotate_90, rotate_180, rotate_270]],
            [[blur]],
        ]

    random_augmenter = RandomAugmenter(
        augmenters, vocab=kwargs.get("vocab", set()), version=version
    )
    return random_augmenter.augment(before_image, after_image, raw_sentences)


# Only change this when new augmentation method is implemented,
# do not delete or comment out working methods
AUGMENTATION_METHODS: List[
    Callable[
        [np.ndarray, np.ndarray, Sequence[str]],
        Tuple[np.ndarray, np.ndarray, Sequence[str]],
    ]
] = [
    horizontal_mirror,
    vertical_mirror,
    left_diagonal_mirror,
    blur,
    rotate_90,
    rotate_180,
    rotate_270,
    brighten_both,
    brighten_before,
    brighten_after,
    random_augment,
]

METHOD_NAME_TO_FUNCTION = {
    k: v
    for (k, v) in [
        (function_.__name__, function_) for function_ in AUGMENTATION_METHODS
    ]
}
