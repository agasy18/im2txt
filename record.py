import tensorflow as tf
from collections import Callable
import inspect

from utlis import progress


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature_list(values):
    return tf.train.FeatureList(feature=[int64_feature(v) for v in values])


def bytes_feature_list(values):
    return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])


def map_dataset_to_record(dataset: tf.data.Dataset, records_path: str, func: Callable, iter_count=None,
                          init_func=None):
    itr = dataset.make_one_shot_iterator().get_next()
    args, *_ = inspect.getfullargspec(func)
    try:
        super_keys = ['i', 'session']
        f = dict([(x, itr[x]) for x in args if x not in super_keys])
    except KeyError:
        raise Exception('invalid argument in passed func allowed args are: ' + ', '.join(list(itr.keys()) + super_keys))
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        init_func and init_func(sess)
        with tf.python_io.TFRecordWriter(records_path) as writer:
            i = 1
            while True:
                try:
                    r_f = sess.run(f)
                except tf.errors.OutOfRangeError as e:
                    print('\nProcessed\n')
                    break
                if 'i' in args and 'i' not in itr:
                    r_f['i'] = i
                if 'session' in args and 'session' not in itr:
                    r_f['session'] = sess
                for e in func(**r_f):
                    writer.write(e.SerializeToString())
                progress(i, iter_count or 100)
                i += 1
    print('\n')
