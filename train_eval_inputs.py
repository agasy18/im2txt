from os import path, rename
import tensorflow as tf
import numpy as np

import image_processing
from utlis import working_dir
from record import map_dataset_to_record, int64_feature, floats_feature

cache_file_name = 'features.tfrecords'
size_file_name = 'features.size'


def input_fn(dataset, feature_extuctor, is_training, cache_dir, batch_size, max_train_epochs):
    with working_dir(cache_dir, True):
        cd = dataset.captions_dataset
        if not path.isfile(cache_file_name):
            create_feature_records(dataset.image_dataset, feature_extuctor, cd, is_training)

    def input_f():
        with working_dir(cache_dir, True):
            fd, feature_size = feature_dataset()

            def merge(f, c):
                with tf.control_dependencies([tf.assert_equal(f['id'], c['image_id'])]):
                    return {
                               'id': f['id'],
                               'features': tf.identity(f['features']),
                               'input_seq': c['input_seq']
                           }, {
                               'target_seq': c['target_seq'],
                               'mask': c['mask']
                           }

            d = tf.data.Dataset.zip((fd, cd)).map(merge, 4)
            if is_training:
                d = d.repeat(max_train_epochs)
                d = d.shuffle(buffer_size=batch_size * 2)
            d = d.padded_batch(batch_size, padded_shapes=({'id': [], 'features': [feature_size], 'input_seq': [None]},
                                                          {'target_seq': [None], 'mask': [None]}))

            def batch_selector(i, t):
                return tf.equal(tf.shape(i['input_seq'])[0], batch_size)

            d = d.filter(batch_selector)

            def batch_reshape(inp, tar):
                for i in [inp, tar]:
                    for key in i:
                        e = i[key]
                        i[key] = tf.reshape(e,
                                            [batch_size] + [s if s is not None else -1 for s in e.shape[1:].as_list()])
                return inp, tar

            d = d.map(batch_reshape).prefetch(10)

            return d.make_one_shot_iterator().get_next()

    return input_f


def feature_dataset():
    with open(size_file_name, 'r') as f:
        feature_size = int(f.readline())

    def _parse(example_serialized):
        features = {
            'width': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'height': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'id': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'features': tf.FixedLenFeature([feature_size], tf.float32)
        }
        return tf.parse_single_example(example_serialized, features)

    return tf.data.TFRecordDataset([path.abspath(cache_file_name)]).map(_parse), feature_size


def create_feature_records(image_dataset: tf.data.Dataset, feature_extuctor, captions_dataset, is_training):
    image_dataset_iter = image_dataset.make_one_shot_iterator().get_next()
    img = image_processing.process_image(image_dataset_iter['jpeg'], is_training=is_training)
    features, *_ = feature_extuctor.build(images=[img],
                                          mode=tf.estimator.ModeKeys.TRAIN if is_training
                                          else tf.estimator.ModeKeys.EVAL,
                                          trainable=False)
    exec_context = {
        'lastImageID': None
    }

    def ext_f(session, image_id):
        if exec_context['lastImageID'] != image_id:
            f, f_id, width, height = session.run([
                features, image_dataset_iter['id'], image_dataset_iter['width'], image_dataset_iter['height']
            ])
            assert (image_id == f_id)
            s_features = {
                'height': int64_feature(height),
                'width': int64_feature(width),
                'id': int64_feature(image_id),
                'features': floats_feature(f.astype(np.float)[0])
            }
            exec_context['f_size'] = f.size
            exec_context['lastImageID'] = image_id
            exec_context['example'] = tf.train.Example(features=tf.train.Features(feature=s_features))

        yield exec_context['example']

    map_dataset_to_record(dataset=captions_dataset,
                          records_path=cache_file_name + '_tmp',
                          func=ext_f,
                          init_func=lambda sess: feature_extuctor.load(sess))
    rename(cache_file_name + '_tmp', cache_file_name)
    with open(size_file_name, 'w') as f:
        f.write(str(exec_context['f_size']))
