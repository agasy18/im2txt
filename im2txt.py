from argparse import ArgumentParser
from os import path, listdir, makedirs
import tensorflow as tf
import numpy as np
import config
import eval_utils
from utlis import call_program

parser = ArgumentParser()
parser.add_argument('mode', choices=['train', 'eval', 'test'])

parser.add_argument('--test_urls', default=None, help=', separated')
parser.add_argument('--model_dir', default='model', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default=None, help="load specified model")

tf.logging.set_verbosity(tf.logging.INFO)

args = parser.parse_args()

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



def test_input_fn():
    return None


def im2txt(features, labels, mode):
    targets, logits, weights = config.seq_generator(features=features['features'],
                                                    input_seq=features['input_seq'],
                                                    target_seq=labels['labels'],
                                                    mask=labels['mask'],
                                                    mode=mode)

    if mode != tf.estimator.ModeKeys.PREDICT:
        total_loss, losses = config.seq_loss(targets=targets, logits=logits, weights=weights)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              eval_metric_ops=eval_utils.eval_perplexity(weights=weights,
                                                                                         losses=losses))
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=total_loss,
                                              train_op=config.optimize_loss(total_loss))
    else:

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions={'coef': logits, 'ides': targets})


estimator = tf.estimator.Estimator(model_fn=im2txt,
                                   model_dir=model_dir(model_name),
                                   config=tf.estimator.RunConfig().replace(
                                       save_checkpoints_steps=config.save_checkpoints_steps,
                                       keep_checkpoint_max=config.keep_checkpoint_max,
                                       log_step_count_steps=config.log_step_count_steps
                                   ))

if args.mode == 'train':
    estimator.train(input_fn=config.train_input_fn)
elif args.mode == 'eval':
    estimator.evaluate(input_fn=lambda: config.eval_input_fn, steps=config.num_examples_per_eval)
else:
    for name, pred in zip(args.test_urls.split(','), estimator.predict(input_fn=test_input_fn)):
        print('![](', name, ')\n',
              '\n'.join('`{} ({})`\n'.format(' '.join([config.caption_vocabulary.id_to_word(i) for i in ides]), coef)
                        for coef, ides in zip(pred['coef'], np.transpose(pred['ides']))))
