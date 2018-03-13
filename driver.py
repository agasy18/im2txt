import os
import subprocess
import sys
import random
from argparse import ArgumentParser

import numpy as np
import time

parser = ArgumentParser()
parser.add_argument('--gen', action='store_true')
parser.add_argument('--model_dir', default='driver', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default=None, help="load specified model")
args = parser.parse_args()

values = [
    # ['batch_size', 32, 64, 512],
    # ['initial_learning_rate'] + list(np.arange(2.0, 7.0, 0.1)),
    # ['learning_rate_decay_factor'] + list(np.arange(0.3, 0.7, 0.01)),
    # ['clip_gradients'] + list(np.arange(2.0, 8.0, 0.1)),
    # ['lstm_dropout_keep_prob'] + list(np.arange(0.3, 0.7, 0.1)),
    # ['features_dropout_keep_prob'] + list(np.arange(0.3, 0.7, 0.1)),
    ['weight_declay'] + list(np.arange(0.01, 0.1, 0.03)) + list(np.arange(0.1, 1, 0.3)) + list(np.arange(1, 10, 3)) + list(np.arange(10, 100, 3))
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
    print('Loading: ' + jp)
    try:
        with open(jp) as f:
            import json
            return json.load(f)
    except:
        return None


def get_metric(info):
    return [i['perplexity'] for i in info]


def overfeeting(m):
    return len(m) > 1 and not sorted(m, reverse=True) == m


def dont_training(m):
    return len(m) > 3 and m[-3] <= m[-2] <= m[-1]


def validate_eval_info(info):
    m = get_metric(info)
    print('Metric array')
    print(m)
    # if overfeeting(m):
    #     print('Overfeeted')
    #     return False
    if dont_training(m):
        print('training not going')
        return False
    return True


def call(var, env):
    if len(var):
        v = var[0]
        env[v[0]] = str(v[random.randrange(1, len(v))])
        call(var[1:], env)
    else:
        if args.gen:
            exec_name = '-'.join('{}:{}'.format(aliases[k], vx) for k, vx in sorted(env.items()))
        else:
            exec_name = time.strftime("%Y%m%d-%H%M%S")
        if args.model_name:
            exec_name = args.model_name
        m_dir = os.path.join(model_dir, exec_name)
        if os.path.isdir(m_dir) and args.gen:
            print('Skip:' + m_dir)
            return
        print('Exec: ' + m_dir)
        process = subprocess.Popen(['python3', 'im2txt.py', 'train-eval',
                                    '--model_dir', model_dir,
                                    '--model_name', exec_name],
                                   env={**env, **os.environ.copy()},
                                   stdin=sys.stdin,
                                   stdout=sys.stdout,
                                   stderr=sys.stderr)

        for i in range(8 * 60):
            try:
                ev = process.wait(60)
                if ev != 0:
                    exit(ev)
                return
            except subprocess.TimeoutExpired:
                j = load_eval_json(m_dir)
                if j is None:
                    print('Can\'t find eval-info.json in:' + m_dir)
                elif validate_eval_info(j):
                    continue
                print('Terminating validate_eval_info ...')
                process.terminate()
                return
        print('Terminating Timeout ...')
        process.terminate()


if args.gen:
    print('Runing inf loop')
    while True:
        call(values, {})
call([], {})
