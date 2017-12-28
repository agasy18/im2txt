import tensorflow as tf
from os import path

from utlis import call_program, working_dir


class FeatureExtractor:
    def name(self):
        return self.__class__.__name__

    def load(self, sess: tf.Session):
        """
        load model if needed

        :param sess: tensorflow session
        """
        raise NotImplementedError()

    def build(self, images: tf.Tensor, mode: str, trainable: bool, **kwargs) -> tf.Tensor:
        """
        build feature extractor for images
        :returns feature extractor tensor for specified parameters

        :param images: batch of rgb images ([B, H, W, C]) normalized in -1 to 1
        :param mode: one of tf.estimator.ModeKeys
        :param trainable:
        :param kwargs:
        """
        raise NotImplementedError()


class DownloadableFeatureExtractor(FeatureExtractor):
    def __init__(self, cache_dir, url, tar, model_file):
        self.url = url
        self.cache_dir = path.abspath(cache_dir)
        self.model_file = model_file
        self.tar = tar
        self._model_path = path.join(self.cache_dir, model_file)


    @property
    def model_path(self):
        if not path.isfile(self._model_path):
            with working_dir(self.cache_dir):
                call_program(['wget', '-nc', self.url])
                call_program(['tar', '-xvf', self.tar, '-C', './'])
        return self._model_path
