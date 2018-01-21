from feature_extractor import DownloadableFeatureExtractor
import tensorflow as tf


class ObjectDetectorFE(DownloadableFeatureExtractor):
    def name(self):
        return self._name

    def __init__(self, cache_dir, url, tar, model_file, feature_selector, name):
        super().__init__(cache_dir, url, tar, model_file)
        self.feature_selector = feature_selector
        self._name = name

    def build(self, images: tf.Tensor, mode: str, trainable: bool, **kwargs):
        self.download()
        od_graph_def = tf.GraphDef()
        images = tf.cast((images + 1.0) * (0.5 * 255), dtype=tf.uint8, name='detector_image')
        with tf.gfile.GFile(self._model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='', input_map={'image_tensor:0': images})
        return tf.reshape(self.feature_selector(), [tf.shape(images)[0], -1]), {
            'detection_scores': tf.get_default_graph().get_tensor_by_name('detection_scores:0'),
            'detection_classes': tf.get_default_graph().get_tensor_by_name('detection_classes:0'),
            'num_detections': tf.get_default_graph().get_tensor_by_name('num_detections:0'),
            'detection_boxes': tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        }

    def output_constructor(self, pred, out_dic):
        out_dic['detection_scores'] = pred['detection_scores'].tolist()
        out_dic['detection_classes'] = pred['detection_classes'].tolist()
        out_dic['num_detections'] = pred['num_detections'].tolist()
        out_dic['detection_boxes'] = pred['detection_boxes'].tolist()

    def load(self, sess: tf.Session):
        pass
