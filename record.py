import tensorflow as tf
from collections import Callable
import inspect
import itertools
from typing import Iterable

from os import rename, makedirs, path

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


def get_records_length(records_path: str):
    with open(records_path + '.length') as f:
        return int(f.readline())


def set_records_length(records_path: str, length: int):
    with open(records_path + '.length', 'w') as f:
        f.write(str(length))


def map_dataset_to_record(dataset: tf.data.Dataset, records_path: str, func: Callable, iter_count=None,
                          init_func=None):
    itr = dataset.make_one_shot_iterator().get_next()
    args, *_ = inspect.getfullargspec(func)
    super_keys = ['i', 'session']
    try:
        f = dict([(x, itr[x]) for x in args if x not in super_keys])
    except KeyError:
        raise Exception('invalid argument in passed func allowed args are: ' + ', '.join(list(itr.keys()) + super_keys))

    def _func(sess, i):
        r_f = sess.run(f)
        if 'i' in args and 'i' not in itr:
            r_f['i'] = i
        if 'session' in args and 'session' not in itr:
            r_f['session'] = sess
        return func(**r_f)

    map_to_record(itertools.count(), records_path, _func, iter_count, init_func)


def map_to_record(data: Iterable, records_path: str, func, iter_count=None,
                  init_func=None):
    with tf.Session() as sess:
        init_func and init_func(sess)
        if path.dirname(records_path):
            makedirs(path.dirname(records_path), exist_ok=True)
        with tf.python_io.TFRecordWriter(records_path + '_tmp') as writer:
            i = 0
            di = 0
            for d in data:
                try:
                    r_f = func(sess, d)
                except tf.errors.OutOfRangeError:
                    break
                for e in r_f:
                    writer.write(e.SerializeToString())
                    di += 1
                i += 1
                progress(i, iter_count or 100)
    rename(records_path + '_tmp', records_path)
    set_records_length(records_path, di)
    print('\nProcessed: {}, writed: {}\n'.format(i, di))
