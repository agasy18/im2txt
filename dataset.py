from abc import ABC, abstractmethod

import tensorflow as tf


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, words):
        self._words = words
        self._vocab = dict((w, i) for i, w in enumerate(words))
        self.unk_id = len(words)

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self._words):
            return '<UNK>'
        else:
            return self._words[word_id]

    def __len__(self):
        return len(self._words) + 1


class Dataset(ABC):
    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary:
        pass

    @property
    @abstractmethod
    def image_dataset(self) -> tf.data.Dataset:
        pass

    @property
    @abstractmethod
    def image_dataset_length(self) -> int:
        pass

    @property
    @abstractmethod
    def captions_dataset(self) -> tf.data.Dataset:
        pass

    @property
    @abstractmethod
    def captions_dataset_length(self) -> int:
        pass
