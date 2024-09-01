import functools
import os
from typing import Callable, List

import cv2 as cv
import numpy as np
import pytest
from PIL import Image

from src.augmentation_methods import (
    blur,
    brighten_after,
    brighten_before,
    brighten_both,
    horizontal_mirror,
    left_diagonal_mirror,
    rotate_90,
    rotate_180,
    rotate_270,
    vertical_mirror,
)

#######################
#                     #
#  Utility functions  #
#                     #
#######################


def load_image(image_path: str) -> cv.typing.MatLike:
    return cv.imread(image_path)


class BaselineImageLoader:
    def __init__(self, path_to_folder_containing_baseline_images: str) -> None:
        """
        Initialize the BaselineImageLoader with the path to the folder containing the baseline images.

        Args:
            path_to_folder_containing_baseline_images (str): The path to the folder containing the baseline images.
        """
        self.baseline_images_folder_path = (
            path_to_folder_containing_baseline_images
        )

    def load_image_from_folder_path(self, image_name) -> cv.typing.MatLike:
        image_path = os.path.join(self.baseline_images_folder_path, image_name)
        return load_image(image_path)

    @functools.lru_cache(maxsize=1)
    def get_base_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(self.get_base_image_name())

    def get_blurred_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(self.get_blurred_image_name())

    def get_brightened_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_brightened_image_name()
        )

    def get_horizontal_mirrored_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_horizontal_mirrored_image_name()
        )

    def get_vertical_mirrored_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_vertical_mirrored_image_name()
        )

    def get_left_diagonal_mirrored_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_left_diagonal_mirrored_image_name()
        )

    def get_rotated_90_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_rotated_90_image_name()
        )

    def get_rotated_180_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_rotated_180_image_name()
        )

    def get_rotated_270_image(self) -> cv.typing.MatLike:
        return self.load_image_from_folder_path(
            self.get_rotated_270_image_name()
        )

    @staticmethod
    def get_base_image_name() -> str:
        return "base.png"

    @staticmethod
    def get_blurred_image_name() -> str:
        return "blurred.png"

    @staticmethod
    def get_brightened_image_name() -> str:
        return "brightened.png"

    @staticmethod
    def get_horizontal_mirrored_image_name() -> str:
        return "horizontal_mirrored.png"

    @staticmethod
    def get_vertical_mirrored_image_name() -> str:
        return "vertical_mirrored.png"

    @staticmethod
    def get_left_diagonal_mirrored_image_name() -> str:
        return "left_diagonal_mirrored.png"

    @staticmethod
    def get_rotated_90_image_name() -> str:
        return "rotated_90.png"

    @staticmethod
    def get_rotated_180_image_name() -> str:
        return "rotated_180.png"

    @staticmethod
    def get_rotated_270_image_name() -> str:
        return "rotated_270.png"


baseline_image_loaders = [
    BaselineImageLoader("tests/baseline_images/happy_dog")
]


def get_base_images():
    return (
        baseline_image.get_base_image()
        for baseline_image in baseline_image_loaders
    )


def get_rotated_90_images():
    return (
        baseline_image.get_rotated_90_image()
        for baseline_image in baseline_image_loaders
    )


def get_rotated_180_images():
    return (
        baseline_image.get_rotated_180_image()
        for baseline_image in baseline_image_loaders
    )


def get_rotated_270_images():
    return (
        baseline_image.get_rotated_270_image()
        for baseline_image in baseline_image_loaders
    )


def get_blurred_images():
    return (
        baseline_image.get_blurred_image()
        for baseline_image in baseline_image_loaders
    )


def get_brightened_images():
    return (
        baseline_image.get_brightened_image()
        for baseline_image in baseline_image_loaders
    )


def get_horizontal_mirrored_images():
    return (
        baseline_image.get_horizontal_mirrored_image()
        for baseline_image in baseline_image_loaders
    )


def get_vertical_mirrored_images():
    return (
        baseline_image.get_vertical_mirrored_image()
        for baseline_image in baseline_image_loaders
    )


def get_left_diagonal_mirrored_images():
    return (
        baseline_image.get_left_diagonal_mirrored_image()
        for baseline_image in baseline_image_loaders
    )


def assert_images_equal(image_1: np.ndarray, image_2: np.ndarray) -> None:
    assert isinstance(image_1, np.ndarray) and isinstance(image_2, np.ndarray)
    convert_opencv_to_pillow_image = lambda image: Image.fromarray(
        cv.cvtColor(image, cv.COLOR_BGR2RGB)
    )

    img1 = convert_opencv_to_pillow_image(image_1)
    img2 = convert_opencv_to_pillow_image(image_2)

    compare_images_equal(img1, img2)


def compare_images_equal(img1: Image.Image, img2: Image.Image) -> None:
    # TODO: find a better way to compare images
    # Convert to same mode and size for comparison
    img2 = img2.convert(img1.mode)
    img2 = img2.resize(img1.size)

    sum_sq_diff = np.sum(
        (np.asarray(img1).astype("float") - np.asarray(img2).astype("float"))
        ** 2
    )

    if sum_sq_diff == 0:
        # Images are exactly the same
        pass
    else:
        normalized_sum_sq_diff = sum_sq_diff / np.sqrt(sum_sq_diff)
        assert normalized_sum_sq_diff < 700  # This is a threshold value


def check_two_lists_of_strings_equal(
    candidates: List[str], references: List[str]
) -> None:
    for candidate_string, reference_string in zip(candidates, references):
        assert candidate_string == reference_string


def augment_single_image_and_check_before_and_after_results(
    aug_method: Callable,
    base_image: cv.typing.MatLike,
    expected_image: cv.typing.MatLike,
) -> None:
    """
    Augment a single image using the given augmentation method and check if the before and after images are as expected.

    Args:
        aug_method (Callable): The augmentation method to apply.
        base_image (cv.typing.MatLike): The base image to augment.
        expected_image (cv.typing.MatLike): The expected image after augmentation.

    Raises:
        AssertionError: If the before and after images are not as expected.
    """
    before_image, after_image, _ = aug_method(
        base_image, base_image, ["some sentence"]
    )

    assert_images_equal(before_image, after_image)
    assert_images_equal(expected_image, after_image)


###########################
#                         #
#  Image Transformations  #
#          Tests          #
#                         #
###########################


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_rotated_90_images()),
)
def test_rotate_90_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        rotate_90, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_rotated_180_images()),
)
def test_rotate_180_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        rotate_180, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_rotated_270_images()),
)
def test_rotate_270_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        rotate_270, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_blurred_images()),
)
def test_blur_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        blur, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_brightened_images()),
)
def test_brighten_both_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        brighten_both, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_brightened_images()),
)
def test_brighten_before_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    before_image, after_image, _ = brighten_before(
        base_image, base_image, ["some sentence"]
    )
    assert_images_equal(
        before_image, expected_image
    )  # Brighten before should change the before image
    assert_images_equal(
        after_image, base_image
    )  # Brighten before should not change the after image


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_brightened_images()),
)
def test_brighten_after_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    before_image, after_image, _ = brighten_after(
        base_image, base_image, ["some sentence"]
    )

    assert_images_equal(
        before_image, base_image
    )  # Brighten after should not change the before image
    assert_images_equal(
        after_image, expected_image
    )  # Brighten after should change the after image


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_horizontal_mirrored_images()),
)
def test_horizontal_mirror_images_augmentation(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        horizontal_mirror, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_vertical_mirrored_images()),
)
def test_vertical_mirror_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        vertical_mirror, base_image, expected_image
    )


@pytest.mark.parametrize(
    "base_image, expected_image",
    zip(get_base_images(), get_left_diagonal_mirrored_images()),
)
def test_left_diagonal_mirror_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    augment_single_image_and_check_before_and_after_results(
        left_diagonal_mirror, base_image, expected_image
    )


@pytest.mark.skip("Not implemented yet")
def test_random_augment_image_transformations(
    base_image: cv.typing.MatLike, expected_image: cv.typing.MatLike
) -> None:
    raise NotImplementedError("Test not implemented yet")


###########################
#                         #
#  Text Transformations   #
#          Tests          #
#                         #
###########################


@pytest.fixture(
    params=(
        "bottom",
        "lower",
        "below",
        "bottommost",
        "lowest",
    )
)
def bottom_direction(request):
    return request.param


@pytest.fixture(
    params=(
        "top",
        "upper",
        "above",
        "topmost",
        "highest",
        "higher",
    )
)
def top_direction(request):
    return request.param


@pytest.fixture(params=("left",))
def left_direction(request):
    return request.param


@pytest.fixture(params=("right",))
def right_direction(request):
    return request.param


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def rotate_90_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, "right"
    elif request.param == "right":
        return right_direction, "bottom"
    elif request.param == "bottom":
        return bottom_direction, "left"
    elif request.param == "left":
        return left_direction, "top"
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            "bottom right",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            "top right",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            "bottom left",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            "top left",
        )
    else:
        raise ValueError("Invalid parameter value")


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def rotate_180_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, "bottom"
    elif request.param == "right":
        return right_direction, "left"
    elif request.param == "bottom":
        return bottom_direction, "top"
    elif request.param == "left":
        return left_direction, "right"
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            "bottom left",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            "bottom right",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            "top left",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            "top right",
        )
    else:
        raise ValueError("Invalid parameter value")


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def rotate_270_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, "left"
    elif request.param == "right":
        return right_direction, "top"
    elif request.param == "bottom":
        return bottom_direction, "right"
    elif request.param == "left":
        return left_direction, "bottom"
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            "top left",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            "bottom left",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            "top right",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            "bottom right",
        )
    else:
        raise ValueError("Invalid parameter value")


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def horizontal_mirror_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, top_direction
    elif request.param == "right":
        return right_direction, "left"
    elif request.param == "bottom":
        return bottom_direction, bottom_direction
    elif request.param == "left":
        return left_direction, "right"
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            f"{top_direction} left",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            f"{top_direction} right",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            f"{bottom_direction} left",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            f"{bottom_direction} right",
        )
    else:
        raise ValueError("Invalid parameter value")


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def vertical_mirror_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, "bottom"
    elif request.param == "right":
        return right_direction, right_direction
    elif request.param == "bottom":
        return bottom_direction, "top"
    elif request.param == "left":
        return left_direction, left_direction
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            f"bottom {right_direction}",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            f"bottom {left_direction}",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            f"top {right_direction}",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            f"top {left_direction}",
        )
    else:
        raise ValueError("Invalid parameter value")


@pytest.fixture(
    params=(
        "top",
        "right",
        "bottom",
        "left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    )
)
def left_diagonal_mirror_text_transformation(
    request, top_direction, right_direction, bottom_direction, left_direction
):
    if request.param == "top":
        return top_direction, "left"
    elif request.param == "right":
        return right_direction, "bottom"
    elif request.param == "bottom":
        return bottom_direction, "right"
    elif request.param == "left":
        return left_direction, "top"
    elif request.param == "top right":
        return (
            f"{top_direction} {right_direction}",
            "bottom left",
        )
    elif request.param == "top left":
        return (
            f"{top_direction} {left_direction}",
            f"{top_direction} {left_direction}",
        )
    elif request.param == "bottom right":
        return (
            f"{bottom_direction} {right_direction}",
            f"{bottom_direction} {right_direction}",
        )
    elif request.param == "bottom left":
        return (
            f"{bottom_direction} {left_direction}",
            "top right",
        )
    else:
        raise ValueError("Invalid parameter value")


def test_rotate_90_text_transformations_on_single_direction(
    dummy_image: np.ndarray, rotate_90_text_transformation
) -> None:
    original_direction, expected_direction = rotate_90_text_transformation

    _, _, augmented_sentences = rotate_90(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])


def test_rotate_180_text_transformations_on_single_direction(
    dummy_image: np.ndarray, rotate_180_text_transformation
) -> None:
    original_direction, expected_direction = rotate_180_text_transformation

    _, _, augmented_sentences = rotate_180(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])


def test_rotate_270_text_transformations_on_single_direction(
    dummy_image: np.ndarray, rotate_270_text_transformation
) -> None:
    original_direction, expected_direction = rotate_270_text_transformation

    _, _, augmented_sentences = rotate_270(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])


def test_horizontal_mirror_text_transformations_on_single_direction(
    dummy_image: np.ndarray, horizontal_mirror_text_transformation
) -> None:
    original_direction, expected_direction = (
        horizontal_mirror_text_transformation
    )

    _, _, augmented_sentences = horizontal_mirror(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])


def test_vertical_mirror_text_transformations_on_single_direction(
    dummy_image: np.ndarray, vertical_mirror_text_transformation
) -> None:
    original_direction, expected_direction = (
        vertical_mirror_text_transformation
    )

    _, _, augmented_sentences = vertical_mirror(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])


def test_left_diagonal_mirror_text_transformations_on_single_direction(
    dummy_image: np.ndarray, left_diagonal_mirror_text_transformation
) -> None:
    original_direction, expected_direction = (
        left_diagonal_mirror_text_transformation
    )

    _, _, augmented_sentences = left_diagonal_mirror(
        dummy_image, dummy_image, [original_direction]
    )

    # Check if the augmented sentences are as expected
    check_two_lists_of_strings_equal(augmented_sentences, [expected_direction])
