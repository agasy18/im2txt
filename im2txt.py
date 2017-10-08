import tensorflow as tf
import model
from os import path, listdir, makedirs
import data_processors
from argparse import ArgumentParser

from utlis import call_program

parser = ArgumentParser()
parser.add_argument('mode', choices=['train', 'eval', 'test'])

parser.add_argument('--test_urls', default='', )
parser.add_argument('--records_dir', default='records', help='directory for records')
parser.add_argument('--model_dir', default='model')
parser.add_argument('--model_index', default=None)
parser.add_argument('--keep_model_max', default=5)

parser.add_argument('--train_prefix', default='train2014')
parser.add_argument('--val_prefix', default='val2014')

parser.add_argument('--image_records', default='images')
parser.add_argument('--vocabulary', default='vocabulary.txt')
parser.add_argument('--features', default='features')

parser.add_argument('--max_train_iters', default=10000000)
parser.add_argument('--keep_checkpoint_max', default=10)
parser.add_argument('--max_train_epochs', default=3)
parser.add_argument('--save_checkpoints_steps', default=10000)
parser.add_argument('--log_step_count_steps', default=10000)
parser.add_argument('--num_examples_per_epoch', default=586363)
parser.add_argument('--num_examples_per_eval', default=40504)

parser.add_argument('--num_lstm_units', default=512)
parser.add_argument('--embedding_size', default=512)
parser.add_argument('--lstm_dropout_keep_prob', default=0.7)
parser.add_argument('--initializer_scale', default=0.08)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--initial_learning_rate', default=2.0)
parser.add_argument('--learning_rate_decay_factor', default=0.5)
parser.add_argument('--num_epochs_per_decay', default=8.0)
parser.add_argument('--optimizer', default='SGD')
parser.add_argument('--clip_gradients', default=5.0)

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
    'vocab_size': len(voc)
}

model_index = args.model_index

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


if model_index is None:
    ms = sorted([int(d) for d in listdir(args.model_dir) if d.isdecimal()])
    if not len(ms):
        model_index = 0
    else:
        model_index = ms[-1] + 1 if args.mode == 'train' else ms[-1]
        [model_dir(d, rm=True) for d in ms[:-args.keep_model_max]]

estimator = tf.estimator.Estimator(model_fn=model.im22txt,
                                   model_dir=model_dir(model_index),
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
    pass


if args.mode == 'train':
    estimator.train(input_fn=lambda: train_eval_input_fn(args.train_prefix))
elif args.mode == 'eval':
    estimator.evaluate(input_fn=lambda: train_eval_input_fn(args.val_prefix), steps=args.num_examples_per_eval)
else:
    pass
