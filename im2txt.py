from argparse import ArgumentParser
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
parser.add_argument('mode', choices=['train', 'eval', 'test'])
parser.add_argument('--debug', action='store_true')

parser.add_argument('--model_dir', default='model', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default=None, help="load specified model")
parser.add_argument('--keep_model_max', type=int, default=5)

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
        model_name = ms[-1] + 1 if args.mode == 'train' else ms[-1]
        [model_dir(d, rm=True) for d in ms[:-args.keep_model_max]]


def test_input_fn():
    return None


def caption_log_fn(id, input_seq, mask, targets):
    s = '[{}] {} | {}'.format(id,
                              ' '.join(config.caption_vocabulary.id_to_word(i) for i in input_seq[:sum(mask)]),
                              ' '.join(config.caption_vocabulary.id_to_word(t) for t in targets)
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
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('caption_log'):
            tf.summary.text('caption', tf.py_func(caption_log_fn, [
                ides[0],
                target_seq[0],
                mask[0],
                tf.argmax(pred['logits'], 1)[0:tf.shape(mask)[1]],
            ], tf.string, stateful=False))

        total_loss, losses = config.seq_loss(targets=target_seq, logits=pred['logits'], mask=mask)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              eval_metric_ops=eval_utils.eval_perplexity(mask=mask,
                                                                                         losses=losses))
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              train_op=config.optimize_loss(total_loss))
    else:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={'coef': pred['coefs'], 'ides': pred['ides']})


estimator = tf.estimator.Estimator(model_fn=im2txt,
                                   model_dir=model_dir(model_name),
                                   config=tf.estimator.RunConfig().replace(
                                       save_checkpoints_steps=config.save_checkpoints_steps,
                                       keep_checkpoint_max=config.keep_checkpoint_max,
                                       log_step_count_steps=config.log_step_count_steps
                                   ))
hooks = []
if args.debug:
    hooks.append(tf_debug.LocalCLIDebugHook())

if args.mode == 'train':
    in_f = config.train_input_fn()
    estimator.train(input_fn=in_f, hooks=hooks)
elif args.mode == 'eval':
    in_f = config.eval_input_fn()
    estimator.evaluate(input_fn=in_f, steps=config.num_examples_per_eval, hooks=hooks)
else:
    for name, pred in zip(args.test_urls.split(','), estimator.predict(input_fn=test_input_fn)):
        print('![](', name, ')\n',
              '\n'.join('`{} ({})`\n'.format(' '.join([config.caption_vocabulary.id_to_word(i) for i in ides]), coef)
                        for coef, ides in zip(pred['coef'], np.transpose(pred['ides']))))
