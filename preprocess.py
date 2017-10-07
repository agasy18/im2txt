from argparse import ArgumentParser
from os import path, makedirs

from data_processors import create_image_records, create_vocab, tokenize_captions, load_vocab, create_captions_records
from utlis import call_program, working_dir, gs_download

parser = ArgumentParser()

parser.add_argument('--data_dir',
                    default='data', help='data dir')

parser.add_argument('--gs_downloads', help='urls separated with ,',
                    default='gs://images.cocodataset.org/train2014,'
                            'gs://images.cocodataset.org/val2014,'
                            'gs://images.cocodataset.org/annotations')

parser.add_argument('--downloads', help='urls separated with ,',
                    default='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')

parser.add_argument('--unzip', default='annotations/annotations_trainval2014.zip', help='paths separated with ,')
parser.add_argument('--tar', default='inception_v3_2016_08_28.tar.gz', help='paths separated with ,')

parser.add_argument('--train_caption_json', default='annotations/captions_train2014.json')
parser.add_argument('--val_caption_json', default='annotations/captions_val2014.json')
parser.add_argument('--train_image_dir', default='train2014')
parser.add_argument('--val_image_dir', default='val2014')
parser.add_argument('--records_dir', default='records')
parser.add_argument('--image_records', default='images')
parser.add_argument('--vocabulary', default='vocabulary.txt')
parser.add_argument('--features', default='features')
parser.add_argument('--min_word_count', default=4)
parser.add_argument('--cnn_model', default='inception_v3.ckpt')
parser.add_argument('--start_word', default='<S>')
parser.add_argument('--end_word', default='</S>')


args = parser.parse_args()

makedirs(args.data_dir, exist_ok=True)
with working_dir(args.data_dir):
    for x in args.gs_downloads.split(','):
        gs_download(x)

    for x in args.downloads.split(','):
        call_program(['wget', '-nc', x])

    for x in args.unzip.split(','):
        call_program(['unzip', '-n', x])

    for x in args.tar.split(','):
        call_program(['tar', '-xvf', x, '-C', './'])

image_records = path.join(args.records_dir, args.image_records)
makedirs(image_records, exist_ok=True)
for image_dir, caption_json in [(args.train_image_dir, args.train_caption_json),
                                (args.val_image_dir, args.val_caption_json)]:

    records_path = path.join(image_records, image_dir + '.tfrecords')
    if not path.isfile(records_path):
        create_image_records(records_path=records_path,
                             image_dir=path.join(args.data_dir, image_dir),
                             caption_json=path.join(args.data_dir, caption_json))
    else:
        print('Skip', records_path)

vocabulary_path = path.join(args.records_dir, args.vocabulary)

_tokenized_captions = {}


def tokenized_captions(caption_file):
    if caption_file not in _tokenized_captions:
        _tokenized_captions[caption_file] = tokenize_captions(path.join(args.data_dir, caption_file), args.start_word, args.end_word)
    return _tokenized_captions[caption_file]


if not path.isfile(vocabulary_path):
    create_vocab(tokenized_captions(args.train_caption_json), vocabulary_path, args.min_word_count)
else:
    print('Skip', vocabulary_path)


vocabulary = load_vocab(vocabulary_path)

features_records = path.join(args.records_dir, args.features)

makedirs(features_records, exist_ok=True)

for image_dir, caption_json in [(args.train_image_dir, args.train_caption_json),
                                (args.val_image_dir, args.val_caption_json)]:

    records_path = path.join(features_records, image_dir + '.tfrecords')
    img_records = path.join(image_records, image_dir + '.tfrecords')

    if not path.isfile(records_path):
        create_captions_records(records_path=records_path,
                                image_records=img_records,
                                captions=tokenized_captions(caption_json),
                                vocabulary=vocabulary,
                                cnn_model=path.join(args.data_dir, args.cnn_model))
    else:
        print('Skip', records_path)

