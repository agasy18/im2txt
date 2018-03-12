import os
import subprocess
import sys
import random
from argparse import ArgumentParser

import numpy as np

parser = ArgumentParser()
parser.add_argument('--gen', action='store_true')
parser.add_argument('--model_dir', default='driver', help="dir for storing generated model files and logs")
args = parser.parse_args()

values = [
    ['batch_size', 32, 64, 512],
    ['initial_learning_rate'] + list(np.arange(2.0, 7.0, 0.1)),
    ['learning_rate_decay_factor'] + list(np.arange(0.3, 0.7, 0.01)),
    ['clip_gradients'] + list(np.arange(2.0, 8.0, 0.1)),
    ['lstm_dropout_keep_prob'] + list(np.arange(0.3, 0.7, 0.1)),
    ['features_dropout_keep_prob'] + list(np.arange(0.3, 0.7, 0.1))
]

aliases = {
    'learning_rate_decay_factor': 'r',
    'lstm_dropout_keep_prob': 'lk',
    'features_dropout_keep_prob': 'fk'
}

for v, *_ in values:
    if v not in aliases:
        aliases[v] = v[:1]

model_dir = args.model_dir


def load_eval_json(dir):
    jp = os.path.join(dir, 'eval-info.json')
    try:
        with open(jp) as f:
            import json
            return json.load(f)
    except:
        return None


def get_metric(info):
    return [i['perplexity'] for i in info]


def overfeeting(m):
    return len(m) > 1 and sorted(m) == m


def dont_training(m):
    return len(m) > 3 and m[-1] - m[-2] <= m[-2] - m[-3] <= 0.003


def validate_eval_info(info):
    m = get_metric(info)
    return not overfeeting(m) and not dont_training(m)


def call(var, env):
    if len(var):
        v = var[0]
        env[v[0]] = str(v[random.randrange(1, len(v))])
        call(var[1:], env)
    else:
        exec_name = '-'.join('{}:{}'.format(aliases[k], vx) for k, vx in sorted(env.items()))
        m_dir = os.path.join(model_dir, exec_name)
        if os.path.isdir(m_dir):
            print('Skip:' + m_dir)
            return
        print('Exec: ' + m_dir)
        process = subprocess.Popen(['python3', 'im2txt.py', 'train-eval', '--model_dir', m_dir],
                                   env={**env, **os.environ.copy()},
                                   stdin=sys.stdin,
                                   stdout=sys.stdout,
                                   stderr=sys.stderr)
        try:
            for i in range(8):
                ev = process.wait(60 * 60)
                if ev != 0:
                    exit(ev)
                j = load_eval_json(m_dir)
                if not j:
                    print('Can\'t find eval-info.json in:' + m_dir)
                elif validate_eval_info(j):
                    continue
                print('Terminating ...')
                process.terminate()
                break
        except subprocess.TimeoutExpired:
            print('Terminating ...')
            process.terminate()


if args.gen:
    while True:
        call(values, {})
call([], {})
