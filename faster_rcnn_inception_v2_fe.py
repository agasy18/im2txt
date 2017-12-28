from feature_extractor import DownloadableFeatureExtractor
import tensorflow as tf


class FasterRCNNInceptionV2(DownloadableFeatureExtractor):
    def __init__(self, cache_dir, url, tar, model_file):
        super().__init__(cache_dir, url, tar, model_file)

    def build(self, images: tf.Tensor, mode: str, trainable: bool, **kwargs) -> tf.Tensor:
        pass

    def load(self, sess: tf.Session):
        pass
