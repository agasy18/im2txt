import json
from collections import Counter

import nltk as nltk
import tensorflow as tf
import tensorflow.contrib as contrib
from os import path, makedirs, mkdir
import image_processing
import image_embedding
import numpy as np

from utlis import progress


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, words):
        self._words = words
        self._vocab = dict((w, i) for i, w in enumerate(words))
        self._unk_id = len(words)

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self._words):
            return '<UNK>'
        else:
            return self._words[word_id]

    def __len__(self):
        return len(self._words) + 1


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def create_image_records(records_path, image_dir, caption_json):
    with open(caption_json) as f:
        j = json.load(f)
        images = j['images']
    print('Write:', records_path)

    encoded_jpeg = tf.placeholder(dtype=tf.string)
    decoded_jpeg = tf.image.decode_jpeg(encoded_jpeg, channels=3)

    def decode_jpeg(session, jpeg):
        image = session.run(decoded_jpeg, feed_dict={encoded_jpeg: jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3

    with tf.Session() as sess, tf.python_io.TFRecordWriter(records_path) as writer:
        for i, img in enumerate(images):
            try:
                with tf.gfile.FastGFile(path.join(image_dir, img['file_name']), "rb") as f:
                    encoded_image = f.read()
                decode_jpeg(sess, encoded_image)
                features = {
                    'jpeg': _bytes_feature(encoded_image),
                    'id': _int64_feature(img['id']),
                    'height': _int64_feature(img['height']),
                    'width': _int64_feature(img['width'])
                }
                writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
                progress(i + 1, len(images))
            except (tf.errors.InvalidArgumentError, AssertionError):
                print("Skip: file with invalid JPEG data: %s" % img['file_name'])


def create_vocab(captions, word_counts_file, min_word_count):
    print("Creating vocabulary.", word_counts_file)
    counter = Counter()
    for cs in captions.values():
        for c in cs:
            counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(word_counts_file, "w") as f:
        f.write("\n".join(["%s %d" % word_count for word_count in word_counts]))


def load_vocab(word_counts_file):
    with tf.gfile.FastGFile(word_counts_file, "r") as f:
        return Vocabulary([x.split(' ')[0] for x in f.readlines()])


def tokenize_captions(captions_file, start_word, end_word):
    with tf.gfile.FastGFile(captions_file, "r") as f:
        caption_data = json.load(f)

    print("Processing captions.", captions_file)

    # Extract the captions. Each image_id is associated with multiple captions.
    id_to_captions = {}
    for annotation in caption_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        id_to_captions.setdefault(image_id, [])
        id_to_captions[image_id].append(caption)

    def caption(img_id):
        return [[start_word] + nltk.tokenize.word_tokenize(c.lower()) + [end_word] for c in id_to_captions[img_id]]

    return dict((img['id'], caption(img['id'])) for img in caption_data["images"])


def parse_image(example_serialized):
    features = {
        "jpeg": tf.FixedLenFeature((), tf.string, default_value=""),
        "id": tf.FixedLenFeature((), tf.int64, default_value=-1),
        "width": tf.FixedLenFeature((), tf.int64, default_value=-1),
        "height": tf.FixedLenFeature((), tf.int64, default_value=-1),
    }
    return tf.parse_single_example(example_serialized, features)


def parse_caption(sequence_example_serialized):
    context, sequence = tf.parse_single_sequence_example(
        sequence_example_serialized,
        context_features={
            'image_id': tf.FixedLenFeature((), tf.int64, default_value=-1),
            'features': tf.FixedLenFeature([2048], tf.float32)
        },
        sequence_features={
            'caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    return context, sequence


def create_captions_records(records_path, image_records, captions, vocabulary, cnn_model):
    tf.reset_default_graph()
    dataset = contrib.data.TFRecordDataset([image_records])
    dataset = dataset.map(parse_image)
    imgage_record = dataset.make_one_shot_iterator().get_next()
    img_jpeg = imgage_record['jpeg']
    img_id = imgage_record['id']
    print('extract features', records_path, '\n')
    jpg_data = tf.placeholder(dtype=tf.string)
    img = image_processing.process_image(jpg_data, 'train' in image_records)
    features = image_embedding.inception_v3([img], False, 'train' in image_records)
    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3"))
    with tf.Session() as sess:
        saver.restore(sess, cnn_model)
        with tf.python_io.TFRecordWriter(records_path) as writer:
            i = 1
            while True:
                try:
                    img_id_a, img_jpeg_a = sess.run([img_id, img_jpeg])
                except tf.errors.OutOfRangeError as e:
                    print('\nProcessed\n')
                    break

                for caption in captions[img_id_a]:
                    caption_ids = [vocabulary.word_to_id(word) for word in caption]

                    img_a, features_a = sess.run([img, features], feed_dict={jpg_data: img_jpeg_a})

                    context = tf.train.Features(feature={
                        "image_id": _int64_feature(img_id_a),
                        "features": _floats_feature(features_a.astype(np.float)[0]),
                    })
                    feature_lists = tf.train.FeatureLists(feature_list={
                        "caption": _bytes_feature_list([w.encode() for w in caption]),
                        "caption_ids": _int64_feature_list(caption_ids)
                    })

                    sequence_example = tf.train.SequenceExample(
                        context=context, feature_lists=feature_lists)

                    writer.write(sequence_example.SerializeToString())

                progress(i, len(captions))
                i += 1

    print('\n')
