import tensorflow as tf
from os import path

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

from utlis import call_program, working_dir

slim = tf.contrib.slim


class Inception:
    __slots__ = "_model_path"

    def __init__(self, cache_dir, url, tar, model_file):
        self.url = url
        self.cache_dir = path.abspath(cache_dir)
        self.model_file = model_file
        self.tar = tar
        self._model_path = path.join(self.cache_dir, model_file)

    @staticmethod
    def inception_v3(images,
                     trainable=True,
                     is_training=True,
                     weight_decay=0.00004,
                     stddev=0.1,
                     dropout_keep_prob=0.8,
                     use_batch_norm=True,
                     batch_norm_params=None,
                     add_summaries=True,
                     scope="InceptionV3"):
        """Builds an Inception V3 subgraph for image embeddings.

      Args:
        images: A float32 Tensor of shape [batch, height, width, channels].
        trainable: Whether the inception submodel should be trainable or not.
        is_training: Boolean indicating training mode or not.
        weight_decay: Coefficient for weight regularization.
        stddev: The standard deviation of the trunctated normal weight initializer.
        dropout_keep_prob: Dropout keep probability.
        use_batch_norm: Whether to use batch normalization.
        batch_norm_params: Parameters for batch normalization. See
          tf.contrib.layers.batch_norm for details.
        add_summaries: Whether to add activation summaries.
        scope: Optional Variable scope.

      Returns:
        end_points: A dictionary of activations from inception_v3 layers.
      """
        # Only consider the inception model to be in training mode if it's trainable.
        is_inception_model_training = trainable and is_training

        if use_batch_norm:
            # Default parameters for batch normalization.
            if not batch_norm_params:
                batch_norm_params = {
                    "is_training": is_inception_model_training,
                    "trainable": trainable,
                    # Decay for the moving averages.
                    "decay": 0.9997,
                    # Epsilon to prevent 0s in variance.
                    "epsilon": 0.001,
                    # Collection containing the moving mean and moving variance.
                    "variables_collections": {
                        "beta": None,
                        "gamma": None,
                        "moving_mean": ["moving_vars"],
                        "moving_variance": ["moving_vars"],
                    }
                }
        else:
            batch_norm_params = None

        if trainable:
            weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            weights_regularizer = None

        with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    trainable=trainable):
                with slim.arg_scope(
                        [slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                    net, end_points = inception_v3_base(images, scope=scope)
                    with tf.variable_scope("logits"):
                        shape = net.get_shape()
                        net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                        net = slim.dropout(
                            net,
                            keep_prob=dropout_keep_prob,
                            is_training=is_inception_model_training,
                            scope="dropout")
                        net = slim.flatten(net, scope="flatten")

        # Add summaries.
        if add_summaries:
            for v in end_points.values():
                tf.contrib.layers.summaries.summarize_activation(v)

        return net

    @property
    def model_path(self):
        if not path.isfile(self._model_path):
            with working_dir(self.cache_dir):
                call_program(['wget', '-nc', self.url])
                call_program(['tar', '-xvf', self.model_path, '-C', './'])
        return self._model_path

    def load(self, sess):
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3"))
        saver.restore(sess, self.model_path)

    def build(self, images, mode, trainable, **kwargs):
        return self.inception_v3(images, trainable=trainable, is_training=tf.estimator.ModeKeys.TRAIN == mode,
                                 **kwargs)




inputs = {}


def train_eval_input_fn(prefix):
    if prefix in inputs:
        return inputs[prefix]

    dataset = tf.contrib.data.TFRecordDataset([path.join(args.records_dir, args.features, prefix + '.tfrecords')])

    def parse_caption(sequence_example_serialized):
        c, s = data_processors.parse_caption(sequence_example_serialized)
        caption = tf.cast(s['caption_ids'], tf.int32)
        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)
        return {'features': c['features'], 'input_seq': input_seq}, {'target_seq': target_seq, 'mask': indicator}

    dataset = dataset.map(parse_caption, num_threads=4, output_buffer_size=args.batch_size * 4)
    if prefix == args.train_prefix:
        dataset = dataset.repeat(args.max_train_epochs)
        dataset = dataset.shuffle(buffer_size=100000)
    dataset = dataset.padded_batch(args.batch_size, padded_shapes=({'features': [2048], 'input_seq': [None]},
                                                                   {'target_seq': [None], 'mask': [None]}))

    def batch_selector(i, t):
        return tf.equal(tf.shape(i['input_seq'])[0], args.batch_size)

    dataset = dataset.filter(batch_selector)

    def batch_reshape(inp, tar):
        for i in [inp, tar]:
            for key in i:
                e = i[key]
                i[key] = tf.reshape(e, [args.batch_size] + [s if s is not None else -1 for s in e.shape[1:].as_list()])
        return inp, tar

    dataset = dataset.map(batch_reshape)

    inputs[prefix] = dataset.make_one_shot_iterator().get_next()
    return inputs[prefix]


# def test_input_fn():
#     import image_processing
#     import urllib.request
#     import tensorflow.contrib.data as tfdata
#     import image_embedding
#
#     if args.test_urls:
#         jpegs = [urllib.request.urlopen(url).read()
#                  for url in args.test_urls.split(',')]
#
#         with tf.Graph().as_default() as g:
#             jpeg = tf.placeholder(dtype=tf.string)
#
#             image = image_processing.process_image(jpeg, False)
#
#             features = image_embedding.inception_v3([image], False, False)
#
#             saver = tf.train.Saver(tf.get_collection(
#                 tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3"))
#             with tf.Session(graph=g) as sess:
#                 saver.restore(sess, args.cnn_model)
#                 features_list = [sess.run(features, feed_dict={jpeg: j}) for j in jpegs]
#
#         dataset = tfdata.Dataset.from_tensor_slices(np.array(features_list))
#
#         return {'features': dataset.make_one_shot_iterator().get_next()}, None
#     else:
#         raise Exception('pass test_urls')