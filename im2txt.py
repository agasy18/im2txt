from argparse import ArgumentParser
from functools import partial
from os import path, listdir, makedirs, environ
import tensorflow as tf
import numpy as np
import time
import sys

import config
import eval_utils
from utlis import call_program, working_dir
from tensorflow.python import debug as tf_debug

parser = ArgumentParser()
parser.add_argument('mode', choices=['train', 'eval', 'train-eval'])
parser.add_argument('--debug', action='store_true')

parser.add_argument('--model_dir', default='model', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default=None, help="load specified model")
parser.add_argument('--keep_model_max', type=int, default=15)

tf.logging.set_verbosity(tf.logging.INFO)

args = parser.parse_args()

model_name = args.model_name

makedirs(args.model_dir, exist_ok=True)


def model_dir(mx, rm=False):
    m = path.join(args.model_dir, str(mx))
    if rm and path.exists(m):
        print('remove', m)
        call_program(['rm', '-rf', m])
        return
    elif path.isdir(m):
        print('load', m)
    else:
        print('create', m)
        makedirs(m)

    project_ignore = [args.model_dir] + config.project_ignore
    bk_dir = path.join(m, 'run', time.strftime("%Y%m%d-%H%M%S"))
    with working_dir(bk_dir, True), open('env', 'w') as ef, open('args', 'w') as af:
        ef.write(str(environ))
        af.write(' '.join(sys.argv))
        af.write(' '.join(sys.argv))
    for d in listdir():
        if d in project_ignore or d[0] == '.':
            continue
        call_program(['cp', '-r', d, bk_dir])

    return m


if model_name is None:
    ms = sorted([int(d) for d in listdir(args.model_dir) if d.isdecimal()])
    if not len(ms):
        model_name = 0
    else:
        model_name = ms[-1] + 1 if 'train' in args.mode else ms[-1]
        [model_dir(d, rm=True) for d in ms[:-args.keep_model_max]]


def caption_log_fn(id, input_seq, mask, targets):
    s = '[{}] {} | {}'.format(id,
                              ' '.join(config.caption_vocabulary().id_to_word(i) for i in input_seq[:sum(mask)]),
                              ' '.join(config.caption_vocabulary().id_to_word(t) for t in targets)
                              )
    tf.logging.info(s)
    return s


def im2txt(features, labels, mode):
    input_seq = features['input_seq']
    target_seq = labels['target_seq']
    mask = labels['mask']
    img_features = features['features']
    ides = features['id']

    with tf.variable_scope('sequence'):
        pred = config.seq_generator(features=img_features,
                                    input_seq=input_seq,
                                    mask=mask,
                                    mode=mode)
    with tf.variable_scope('caption_log'):
        tf.summary.text('caption', tf.py_func(caption_log_fn, [
            ides[0],
            target_seq[0],
            mask[0],
            tf.argmax(pred['logits'], 1)[0:tf.shape(mask)[1]],
        ], tf.string, stateful=False))

    batch_loss, losses = config.seq_loss(targets=target_seq, logits=pred['logits'], mask=mask)

    tf.summary.scalar('loss/sequence', batch_loss)
    tf.losses.add_loss(batch_loss)

    weight_declay_loss = tf.reduce_mean([tf.nn.l2_loss(v) / tf.to_float(tf.size(v))
                                         for v in tf.trainable_variables()],
                                        name='weight_declay_loss_abs')

    tf.summary.scalar('loss/weight_declay_abs', weight_declay_loss)
    weight_declay_loss = tf.multiply(weight_declay_loss, config.weight_declay, name='weight_declay_loss')
    tf.summary.scalar('loss/weight_declay', weight_declay_loss)
    if config.weight_declay > 0:
        tf.losses.add_loss(weight_declay_loss, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)

    total_loss = tf.losses.get_total_loss()

    tf.summary.scalar('loss/total_loss', total_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          eval_metric_ops=eval_utils.eval_perplexity(mask=mask,
                                                                                     losses=losses))
    else:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          train_op=config.optimize_loss(total_loss))


model_dir_path = model_dir(model_name)

estimator_train = tf.estimator.Estimator(model_fn=im2txt,
                                         model_dir=model_dir_path,
                                         config=tf.estimator.RunConfig().replace(
                                             save_checkpoints_steps=config.save_checkpoints_steps,
                                             keep_checkpoint_max=config.keep_checkpoint_max,
                                             log_step_count_steps=config.train_log_step_count_steps
                                         )) if 'train' in args.mode else None
estimator_eval = tf.estimator.Estimator(model_fn=im2txt,
                                        model_dir=model_dir_path,
                                        config=tf.estimator.RunConfig().replace(
                                            save_checkpoints_steps=config.save_checkpoints_steps,
                                            keep_checkpoint_max=config.keep_checkpoint_max,
                                            log_step_count_steps=config.eval_log_step_count_steps
                                        )) if 'eval' in args.mode else None


def eval_input_fn():
    return config.train_eval_inputs.input_fn(
        dataset=config.eval_dataset,
        image_preprocessor=config.image_preprocessor,
        feature_extractor=config.feature_detector,
        is_training=False,
        cache_dir=config.eval_dataset.records_dir,
        batch_size=config.batch_size,
        max_epochs=1000000)


def train_input_fn():
    return config.train_eval_inputs.input_fn(
        dataset=config.train_dataset,
        image_preprocessor=config.image_preprocessor,
        feature_extractor=config.feature_detector,
        is_training=True,
        cache_dir=config.train_dataset.records_dir,
        batch_size=config.batch_size,
        max_epochs=config.max_train_epochs)


hooks = []

last_eval_results = []

eval_info_path = path.join(model_dir_path, 'eval-info.json')


def store_eval(data):
    global last_eval_results
    if len(last_eval_results) == 0 and path.isfile(eval_info_path):
        import json
        with open(eval_info_path) as f:
            last_eval_results = json.load(f)
    if data:
        last_eval_results.append(dict((k, float(v)) for k, v in data.items()))
    with open(eval_info_path, 'w') as f:
        import json
        json.dump(last_eval_results, f, indent=2)


store_eval(None)

if args.debug:
    hooks.append(tf_debug.LocalCLIDebugHook())

if args.mode == 'train':
    in_f = train_input_fn()
    estimator_train.train(input_fn=in_f, hooks=hooks + config.train_hooks)
elif args.mode == 'eval':
    ch = None
    in_f = eval_input_fn()
    while True:
        if ch == estimator_eval.latest_checkpoint():
            tf.logging.info('waiting for checkpoint')
            time.sleep(10)
            continue
        ch = estimator_eval.latest_checkpoint()
        tf.logging.info('loading checkpoint: ' + ch)
        e_data = estimator_eval.evaluate(input_fn=in_f, steps=config.num_examples_per_eval() / config.batch_size,
                                         hooks=hooks + config.eval_hooks)
        store_eval(e_data)
elif args.mode == 'train-eval':
    train_in = train_input_fn()
    eval_in = eval_input_fn()
    ch = None
    while True:
        estimator_train.train(input_fn=train_in, steps=config.save_checkpoints_steps * config.eval_every_chackpoint,
                              hooks=hooks + config.train_hooks)
        if ch == estimator_train.latest_checkpoint():
            break
        ch = estimator_train.latest_checkpoint()
        tf.logging.info('loading checkpoint: ' + ch)
        e_data = estimator_eval.evaluate(input_fn=eval_in, steps=config.num_examples_per_eval() / config.batch_size,
                                         hooks=hooks + config.eval_hooks)
        store_eval(e_data)
