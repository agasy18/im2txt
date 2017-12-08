import json
from collections import Counter
from os import path, rename

import tensorflow as tf
import tensorflow.contrib as contrib

from utlis import call_program, working_dir, gs_download, bytes_feature_list, int64_feature, \
    int64_feature_list, progress, bytes_feature, map_dataset_to_record


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


class MSCoco:
    end_word = '</S>'
    start_word = '<S>'
    min_word_count = 4

    def __init__(self,
                 cache_dir,
                 images_gs_url,
                 annotations_gs_url,
                 caption_json_path,
                 annotations_zip,
                 image_dir,
                 start_word='<S>',
                 end_word='</S>',
                 min_word_count=4,
                 vocabulary_file='vocabulary.txt',
                 records_dir='mscoco'):

        self.min_word_count = min_word_count
        self.end_word = end_word
        self.start_word = start_word
        self.annotations_gs_url = annotations_gs_url
        self.images_gs_url = images_gs_url
        self.is_downloaded = False
        self.cache_dir = path.abspath(cache_dir)
        self.records_dir = path.join(self.cache_dir, records_dir)
        self.image_dir = path.join(self.cache_dir, image_dir)
        self.vocabulary_file = path.join(self.records_dir, vocabulary_file)
        self.annotations_zip = annotations_zip
        self.caption_json_path = path.join(self.cache_dir, caption_json_path)
        self.images_records_path = path.join(self.records_dir, image_dir + '.images.tfrecords')
        self.caption_records_path = path.join(self.records_dir, image_dir + '.captions.tfrecords')
        self._vocabulary = None

    def _create_vocab(self, captions):
        print("Creating vocabulary.", self.vocabulary_file)
        counter = Counter()
        for cs in captions.values():
            for c in cs:
                counter.update(c)
        print("Total words:", len(counter))

        # Filter uncommon words and sort by descending count.
        word_counts = [x for x in counter.items() if x[1] >= self.min_word_count]
        word_counts.sort(key=lambda x: x[1], reverse=True)
        print("Words in vocabulary:", len(word_counts))

        # Write out the word counts file.
        with tf.gfile.FastGFile(self.vocabulary_file, "w") as f:
            f.write("\n".join(["%s %d" % word_count for word_count in word_counts]))

    def load_vocab(self):
        with tf.gfile.FastGFile(self.vocabulary_file, "r") as f:
            return Vocabulary([x.split(' ')[0] for x in f.readlines()])

    def _tokenize_captions(self):
        import nltk
        print("Processing captions.", self.caption_json_path)

        # Extract the captions. Each image_id is associated with multiple captions.
        id_to_captions = {}

        caption_data = self.caption_json

        for annotation in caption_data["annotations"]:
            image_id = annotation["image_id"]
            caption = annotation["caption"]
            id_to_captions.setdefault(image_id, [])
            id_to_captions[image_id].append(caption)

        def caption(img_id):
            return [[self.start_word] + nltk.tokenize.word_tokenize(c.lower()) + [self.end_word]
                    for c in id_to_captions[img_id]]

        return dict((img['id'], caption(img['id'])) for img in caption_data["images"])

    @property
    def caption_json(self):
        if not path.isfile(self.caption_json_path):
            self._download_annotations()
        with tf.gfile.FastGFile(self.caption_json_path, "r") as f:
            return json.load(f)

    @property
    def vocabulary(self) -> Vocabulary:
        if not self._vocabulary:
            if not path.isfile(self.vocabulary_file):
                self._create_vocab(self._tokenize_captions())
            self._vocabulary = self.load_vocab()
        return self._vocabulary

    def _download_annotations(self):
        with working_dir(self.cache_dir, True):
            gs_download(self.annotations_gs_url)
            call_program(['unzip', '-n', self.annotations_zip])

    def _download_images(self):
        with working_dir(self.cache_dir, True):
            if self.is_downloaded:
                return
            gs_download(self.images_gs_url)
            self.is_downloaded = True

    def create_image_records(self):
        print('Write:', self.images_records_path)
        self._download_images()
        images = self.caption_json['images']
        encoded_jpeg = tf.placeholder(dtype=tf.string)
        decoded_jpeg = tf.image.decode_jpeg(encoded_jpeg, channels=3)

        def decode_jpeg(session, jpeg):
            image = session.run(decoded_jpeg, feed_dict={encoded_jpeg: jpeg})
            assert len(image.shape) == 3
            assert image.shape[2] == 3

        with tf.Session() as sess, tf.python_io.TFRecordWriter(self.images_records_path + '_tmp') as writer:
            for i, img in enumerate(images):
                try:
                    with tf.gfile.FastGFile(path.join(image_dir, img['file_name']), "rb") as f:
                        encoded_image = f.read()
                    decode_jpeg(sess, encoded_image)
                    features = {
                        'jpeg': bytes_feature(encoded_image),
                        'id': int64_feature(img['id']),
                        'height': int64_feature(img['height']),
                        'width': int64_feature(img['width'])
                    }
                    writer.write(tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString())
                    progress(i + 1, len(images))
                except (tf.errors.InvalidArgumentError, AssertionError):
                    print("Skip: file with invalid JPEG data: %s" % img['file_name'])
        rename(self.images_records_path + '_tmp', self.images_records_path)

    @property
    def image_dataset(self) -> contrib.data.Dataset:
        if not path.isfile(self.images_records_path):
            self.create_image_records()

        def _parse_image(example_serialized):
            features = {
                "jpeg": tf.FixedLenFeature((), tf.string, default_value=""),
                "id": tf.FixedLenFeature((), tf.int64, default_value=-1),
                "width": tf.FixedLenFeature((), tf.int64, default_value=-1),
                "height": tf.FixedLenFeature((), tf.int64, default_value=-1),
            }
            return tf.parse_single_example(example_serialized, features)

        return contrib.data.TFRecordDataset([self.images_records_path]).map(_parse_image)

    @property
    def captions_dataset(self) -> contrib.data.Dataset:
        if not path.isfile(self.caption_records_path):
            self.create_captions_records()

        def parse_caption(sequence_example_serialized):
            context, sequence = tf.parse_single_sequence_example(
                sequence_example_serialized,
                context_features={
                    'image_id': tf.FixedLenFeature((), tf.int64, default_value=-1)
                },
                sequence_features={
                    'caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                })
            caption = tf.cast(sequence['caption_ids'], tf.int32)
            caption_length = tf.shape(caption)[0]
            input_length = tf.expand_dims(tf.subtract(caption_length, 1), 0)

            input_seq = tf.slice(caption, [0], input_length)
            target_seq = tf.slice(caption, [1], input_length)
            indicator = tf.ones(input_length, dtype=tf.int32)
            return {
                       'image_id': context['image_id'],
                       'input_seq': input_seq,
                       'target_seq': target_seq,
                       'mask': indicator
                   }

        return contrib.data.TFRecordDataset([self.caption_records_path]).map(parse_caption)

    def create_captions_records(self):
        print('create_captions_records', self.caption_json_path, '\n')

        captions = self._tokenize_captions()
        vocabulary = self.vocabulary

        def mapper(id):
            for caption in captions[id]:
                caption_ids = [vocabulary.word_to_id(word) for word in caption]

                context = tf.train.Features(feature={
                    "image_id": int64_feature(id),
                })
                feature_lists = tf.train.FeatureLists(feature_list={
                    "caption": bytes_feature_list([w.encode() for w in caption]),
                    "caption_ids": int64_feature_list(caption_ids)
                })

                sequence_example = tf.train.SequenceExample(
                    context=context, feature_lists=feature_lists)
                yield sequence_example

        map_dataset_to_record(
            dataset=self.image_dataset,
            records_path=self.caption_records_path + '_tmp',
            func=mapper,
            iter_count=len(captions)
        )

        rename(self.caption_records_path + '_tmp', self.caption_records_path)
