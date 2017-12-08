import mscoco
import inception
import beam_search
import train_utils
import train_eval_inputs
from functools import partial

import feature2seq

keep_model_max = 5

max_train_iters = 10000000
keep_checkpoint_max = 10
max_train_epochs = 80
save_checkpoints_steps = 3000
log_step_count_steps = 1000
num_examples_per_epoch = 586363
num_examples_per_eval = 40504

batch_size = 32
initial_learning_rate = 2.0
learning_rate_decay_factor = 0.5
num_epochs_per_decay = 8.0
optimizer = 'SGD'
clip_gradients = 5.0
seq_max_len = 100
beam_size = 1
num_lstm_units = 512

initializer_scale = 0.08
embedding_size = 512
lstm_dropout_keep_prob = 0.7

data_dir = 'data'
feature_detector_data_dir = 'data/inception'

train_dataset = mscoco.MSCoco(cache_dir=data_dir,
                              images_gs_url='gs://images.cocodataset.org/train2014',
                              annotations_gs_url='gs://images.cocodataset.org/annotations',
                              caption_json_path='annotations/captions_train2014.json',
                              annotations_zip='annotations/annotations_trainval2014.zip',
                              image_dir='train2014')

eval_dataset = mscoco.MSCoco(cache_dir=data_dir,
                             images_gs_url='gs://images.cocodataset.org/val2014',
                             annotations_gs_url='gs://images.cocodataset.org/annotations',
                             caption_json_path='annotations/captions_val2014.json',
                             annotations_zip='annotations/annotations_trainval2014.zip',
                             image_dir='val2014')

caption_vocabulary = train_dataset.vocabulary

vocab_size = len(caption_vocabulary)

feature_detector = inception.Inception(cache_dir=data_dir,
                                       url='http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                                       tar='inception_v3_2016_08_28.tar.gz',
                                       model_file='inception_v3.ckpt')

predictor = partial(beam_search.beam_search,
                    beam_size=beam_size,
                    vocab_size=vocab_size,
                    start_word_index=caption_vocabulary.word_to_id(train_dataset.start_word),
                    end_word_index=caption_vocabulary.word_to_id(train_dataset.end_word),
                    seq_max_len=seq_max_len)

seq_generator = partial(feature2seq.feature2seq,
                        vocab_size=vocab_size,
                        predictor=predictor,
                        initializer_scale=initializer_scale,
                        embedding_size=embedding_size,
                        num_lstm_units=num_lstm_units,
                        lstm_dropout_keep_prob=lstm_dropout_keep_prob)

seq_loss = train_utils.seq_loss

optimize_loss = partial(train_utils.optimize_loss,
                        initial_learning_rate=initial_learning_rate,
                        num_examples_per_epoch=num_examples_per_epoch,
                        num_epochs_per_decay=num_epochs_per_decay,
                        learning_rate_decay_factor=learning_rate_decay_factor,
                        clip_gradients=clip_gradients,
                        batch_size=batch_size,
                        optimizer=optimizer)

eval_input_fn = partial(train_eval_inputs.input_fn,
                        dataset=eval_dataset,
                        feature_extuctor=feature_detector,
                        is_training=True,
                        cache_dir=feature_detector_data_dir,
                        batch_size=batch_size,
                        max_train_epochs=max_train_epochs)

train_input_fn = partial(train_eval_inputs.input_fn,
                         dataset=train_dataset,
                         feature_extuctor=feature_detector,
                         is_training=True,
                         cache_dir=feature_detector_data_dir,
                         batch_size=batch_size,
                         max_train_epochs=max_train_epochs)