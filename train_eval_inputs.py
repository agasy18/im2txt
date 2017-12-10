from os import path, rename
import tensorflow as tf
import numpy as np

import image_processing
from utlis import working_dir, map_dataset_to_record, int64_feature, floats_feature

cache_file_name = 'features.tfrecords'
size_file_name = 'features.size'


def input_fn(dataset, feature_extuctor, is_training, cache_dir, batch_size, max_train_epochs):
    with working_dir(cache_dir, True):
        if not path.isfile(cache_file_name):
            create_feature_records(dataset.image_dataset, feature_extuctor, is_training)

        fd, feature_size = feature_dataset()
        cd = dataset.captions_dataset()

        def merge(e):
            f, c = e
            return {
                       'features': f['features'],
                       'input_seq': c['input_seq']
                   }, {
                       'target_seq': c['target_seq'],
                       'mask': c['mask']
                   }

        d = tf.data.Dataset.zip(fd, cd).map(merge, num_threads=4, output_buffer_size=batch_size * 4)
        if is_training:
            d = d.repeat(max_train_epochs)
            d = d.shuffle(buffer_size=100000)
        d = d.padded_batch(batch_size, padded_shapes=({'features': [feature_size], 'input_seq': [None]},
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

        d = d.map(batch_reshape)

        return d.make_one_shot_iterator().get_next()


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

    return tf.data.TFRecordDataset([cache_file_name]).map(_parse), feature_size


def create_feature_records(dataset, feature_extuctor, is_training):
    jpg_data = tf.placeholder(dtype=tf.string)
    img = image_processing.process_image(jpg_data, is_training=is_training)
    features, *_ = feature_extuctor.build(images=[img],
                                          mode=tf.estimator.ModeKeys.TRAIN if is_training
                                          else tf.estimator.ModeKeys.EVAL,
                                          trainable=False)
    int_box = [0]

    def ext_f(session, id, jpeg, height, width):
        f = session.run(features, feed_dict={jpg_data: jpeg})
        s_features = {
            'height': int64_feature(height),
            'width': int64_feature(width),
            'id': int64_feature(id),
            'features': floats_feature(f.astype(np.float)[0])
            # 'img': floats_feature(i.astype(np.float))
        }
        int_box[0] = f.size
        yield tf.train.Example(features=tf.train.Features(feature=s_features))

    map_dataset_to_record(dataset=dataset,
                          records_path=cache_file_name + '_tmp',
                          func=ext_f,
                          init_func=lambda sess: feature_extuctor.load(sess))
    rename(cache_file_name + '_tmp', cache_file_name)
    with open(size_file_name, 'w') as f:
        f.write(str(int_box[0]))
