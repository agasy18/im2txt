from argparse import ArgumentParser
from os import path, listdir, makedirs
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import OPTIMIZER_CLS_NAMES

import data_processors
import model
from utlis import call_program

parser = ArgumentParser()
parser.add_argument('mode', choices=['train', 'eval', 'test'])

parser.add_argument('--test_urls', default=None, help=', separated')
parser.add_argument('--records_dir', default='records', help='directory for records')
parser.add_argument('--model_dir', default='model', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default=None, help="load specified model")
parser.add_argument('--keep_model_max', type=int, default=5)

parser.add_argument('--train_prefix', default='train2014')
parser.add_argument('--val_prefix', default='val2014')

parser.add_argument('--cnn_model', default='data/inception_v3.ckpt')
parser.add_argument('--image_records', default='images')
parser.add_argument('--vocabulary', default='vocabulary.txt')
parser.add_argument('--features', default='features')

parser.add_argument('--max_train_iters', type=int, default=10000000)
parser.add_argument('--keep_checkpoint_max', type=int, default=10)
parser.add_argument('--max_train_epochs', type=int, default=80)
parser.add_argument('--save_checkpoints_steps', type=int, default=3000)
parser.add_argument('--log_step_count_steps', type=int, default=1000)
parser.add_argument('--num_examples_per_epoch', type=int, default=586363)
parser.add_argument('--num_examples_per_eval', type=int, default=40504)

parser.add_argument('--num_lstm_units', type=int, default=512)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--lstm_dropout_keep_prob', type=float, default=0.7)
parser.add_argument('--initializer_scale', type=float, default=0.08)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--initial_learning_rate', type=float,  default=2.0)
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.5)
parser.add_argument('--num_epochs_per_decay', type=float, default=8.0)
parser.add_argument('--optimizer', default='SGD', choices=list(OPTIMIZER_CLS_NAMES.keys()))
parser.add_argument('--clip_gradients', type=float, default=5.0)
parser.add_argument('--start_word', default='<S>')
parser.add_argument('--end_word', default='</S>')
parser.add_argument('--seq_max_len', type=int, default=100)
parser.add_argument('--beam_size', type=int, default=1)

tf.logging.set_verbosity(tf.logging.INFO)

args = parser.parse_args()

voc = data_processors.load_vocab(path.join(args.records_dir, args.vocabulary))

model_params = {
    'num_lstm_units': args.num_lstm_units,
    'embedding_size': args.embedding_size,
    'lstm_dropout_keep_prob': args.lstm_dropout_keep_prob,
    'initializer_scale': args.initializer_scale,
    'initial_learning_rate': args.initial_learning_rate,
    'num_examples_per_epoch': args.num_examples_per_epoch,
    'batch_size': args.batch_size,
    'num_epochs_per_decay': args.num_epochs_per_decay,
    'learning_rate_decay_factor': args.learning_rate_decay_factor,
    'optimizer': args.optimizer,
    'clip_gradients': args.clip_gradients,
    'vocab_size': len(voc),
    'end_word_index': voc.word_to_id(args.end_word),
    'start_word_index': voc.word_to_id(args.start_word),
    'seq_max_len': args.seq_max_len,
    'beam_size': args.beam_size
}

model_name = args.model_name

makedirs(args.model_dir, exist_ok=True)


def model_dir(mx, rm=False):
    m = path.join(args.model_dir, str(mx))
    if rm:
        print('remove', m)
        call_program(['rm', '-rf', m])
    elif path.isdir(m):
        print('load', m)
    else:
        print('create', m)
        makedirs(m)
    return m


if model_name is None:
    ms = sorted([int(d) for d in listdir(args.model_dir) if d.isdecimal()])
    if not len(ms):
        model_name = 0
    else:
        model_name = ms[-1] + 1 if args.mode == 'train' else ms[-1]
        [model_dir(d, rm=True) for d in ms[:-args.keep_model_max]]

estimator = tf.estimator.Estimator(model_fn=model.im22txt,
                                   model_dir=model_dir(model_name),
                                   params=model_params,
                                   config=tf.estimator.RunConfig().replace(
                                       save_checkpoints_steps=args.save_checkpoints_steps,
                                       keep_checkpoint_max=args.keep_checkpoint_max,
                                       log_step_count_steps=args.log_step_count_steps
                                   ))

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
    dataset = dataset.repeat(1000000 if prefix == args.val_prefix else args.max_train_epochs)
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


def test_input_fn():
    import image_processing
    import urllib.request
    import tensorflow.contrib.data as tfdata
    import image_embedding

    if args.test_urls:
        jpegs = [urllib.request.urlopen(url).read()
                 for url in args.test_urls.split(',')]

        with tf.Graph().as_default() as g:
            jpeg = tf.placeholder(dtype=tf.string)

            image = image_processing.process_image(jpeg, False)

            features = image_embedding.inception_v3([image], False, False)

            saver = tf.train.Saver(tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3"))
            with tf.Session(graph=g) as sess:
                saver.restore(sess, args.cnn_model)
                features_list = [sess.run(features, feed_dict={jpeg: j}) for j in jpegs]

        dataset = tfdata.Dataset.from_tensor_slices(np.array(features_list))

        return {'features': dataset.make_one_shot_iterator().get_next()}, None
    else:
        raise Exception('pass test_urls')


if args.mode == 'train':
    estimator.train(input_fn=lambda: train_eval_input_fn(args.train_prefix))
elif args.mode == 'eval':
    estimator.evaluate(input_fn=lambda: train_eval_input_fn(args.val_prefix), steps=args.num_examples_per_eval)
else:
    for name, pred in zip(args.test_urls.split(','), estimator.predict(input_fn=test_input_fn)):
        print('![](', name, ')\n', '\n'.join('`{} ({})`\n'.format(' '.join([voc.id_to_word(i) for i in ides]), coef)
              for coef, ides in zip(pred['coef'], np.transpose(pred['ides']))))
