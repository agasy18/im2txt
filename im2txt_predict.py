import json
from argparse import ArgumentParser
import tensorflow as tf
from os import path
import sys
import config

parser = ArgumentParser()

parser.add_argument('--model_dir', default='model', help="dir for storing generated model files and logs")
parser.add_argument('--model_name', default='0', help="load specified model")
parser.add_argument('--jpeg-files', default=None)
parser.add_argument('--feature-size', default=2048)
parser.add_argument('--output-type', default='json', choices=['json'])
parser.add_argument('--output-file', default=None)
parser.add_argument('--export-graph', default=None)

tf.logging.set_verbosity(tf.logging.INFO)


def predict_fn(features, mode, params):
    assert (mode == tf.estimator.ModeKeys.PREDICT)

    dfeatures, *others = config.feature_detector.build(features['image_tensor'], mode, False)

    dfeatures.set_shape([1, params['feature-size']])
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    with tf.Session(config=session_config) as sess:
        config.feature_detector.load(sess)

    with tf.variable_scope('sequence'):
        pred = config.seq_generator(features=dfeatures,
                                    input_seq=None,
                                    mask=None,
                                    mode=mode)
    if len(others):
        pred.update(others[0])
    return tf.estimator.EstimatorSpec(mode=mode, predictions=pred, export_outputs={
        'output': tf.estimator.export.PredictOutput(dict(pred))
    })


def main():
    args = parser.parse_args()

    outputs = []

    estimator = tf.estimator.Estimator(model_fn=predict_fn,
                                       model_dir=path.join(args.model_dir, args.model_name),
                                       params={
                                           'feature-size': args.feature_size
                                       })
    if args.export_graph:
        def serving_input_fn():
            inputs = {'image_tensor': tf.placeholder(tf.float32, [config.image_size, config.image_size, 3],
                                                     name='image_tensor')}
            return tf.estimator.export.ServingInputReceiver(inputs, inputs)
        estimator.export_savedmodel(args.export_graph, serving_input_fn)

    if args.jpeg_files:
        jpeg_files = [x.strip() for x in args.jpeg_files.split(',')]
        for j in jpeg_files:
            outputs.append({
                'input-file': j
            })
    else:
        jpeg_files = None
        raise Exception("No input provided")

    def input_fn():
        p = tf.data.Dataset.from_tensor_slices(jpeg_files).make_one_shot_iterator().get_next()
        return {'image_tensor': tf.expand_dims(config.image_preprocessor(tf.read_file(p), is_training=False), 0)}

    predictions = estimator.predict(input_fn=input_fn)

    for out, pred in zip(outputs, predictions):
        config.output_constructor(pred, out)

    json.dump(outputs, open(args.output_file, 'w') if args.output_file else sys.stdout,
              indent=4, check_circular=False)


if __name__ == '__main__':
    main()
