import random
from copy import deepcopy
from typing import Callable, List, Sequence

import numpy as np


class RandomAugmenter:
    def __init__(
        self,
        augmenter_list: List[List[List[Callable]]],
        **kwargs,
    ) -> None:
        self.AUGMENTERS = augmenter_list
        self.MINIMUM_AUGMENTER_COUNT = 1
        self.N_AUGMENTER = len(self.AUGMENTERS)
        self.vocab = kwargs.get("vocab", None)

    def __print_shuffled_augmenters(self, shuffled_augmenters):
        print(f"{__name__} augmenters: ", end="")

        for augmenter in shuffled_augmenters:
            print(augmenter.__name__, end=" ")

        print("")

    def __select_from_categories(self):
        single_aug_list = []
        for aug_cat_list in self.AUGMENTERS:
            aug_cat_list_len = len(aug_cat_list)
            rand_idx = random.randint(0, aug_cat_list_len - 1)
            aug_list = aug_cat_list[rand_idx]
            aug_list_len = len(aug_list)
            rand_idx = random.randint(0, aug_list_len - 1)
            single_aug_list.append(aug_list[rand_idx])
        return single_aug_list

    def __shuffle(self, augmenters):
        random.shuffle(augmenters)

    def __extract_subset(self, augmenters, *, subset_size=2):
        return augmenters[:subset_size]

    def __apply_augmenters(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        raw_sentences: Sequence[str],
        augmenters: Sequence[Callable],
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        augmented_pair = {
            "before": deepcopy(before_image),
            "after": deepcopy(after_image),
            "sentences": deepcopy(raw_sentences),
        }

        for i, augmenter in enumerate(augmenters):
            (
                augmented_pair["before"],
                augmented_pair["after"],
                augmented_pair["sentences"],
            ) = augmenter(
                augmented_pair["before"],
                augmented_pair["after"],
                augmented_pair["sentences"],
                vocab=self.vocab,
            )

        return (
            augmented_pair["before"],
            augmented_pair["after"],
            augmented_pair["sentences"],
        )

    def augment_v3(self, before_image, after_image, raw_sentences):
        shuffled_augmenters = self.__select_from_categories()
        self.__shuffle(shuffled_augmenters)

        shuffled_augmenters = self.__extract_subset(
            shuffled_augmenters, subset_size=random.randint(1, 2)
        )

        self.__print_shuffled_augmenters(shuffled_augmenters)

        return self.__apply_augmenters(
            before_image, after_image, raw_sentences, shuffled_augmenters
        )

    def augment(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        raw_sentences: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        shuffled_augmenters = self.__select_from_categories()
        self.__shuffle(shuffled_augmenters)
        subset_size = random.randint(1, self.N_AUGMENTER)
        shuffled_augmenters = shuffled_augmenters[:subset_size]
        self.__print_shuffled_augmenters(shuffled_augmenters)
        return self.__apply_augmenters(
            before_image, after_image, raw_sentences, shuffled_augmenters
        )
