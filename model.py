import tensorflow as tf
import tensorflow.contrib as contrib

from LSTMRCell import LSTMRCell


def im22txt(features, labels, mode, params):
    initializer = tf.random_uniform_initializer(
        minval=-params['initializer_scale'],
        maxval=params['initializer_scale'])

    # Image Embedding
    with tf.variable_scope("image_embedding") as scope:
        image_embeddings = contrib.layers.fully_connected(
            inputs=features['features'],
            num_outputs=params['embedding_size'],
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=None,
            scope=scope)

    # Seq Embedding
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name="map",
            shape=[params['vocab_size'], params['embedding_size']],
            initializer=initializer)
        if mode != tf.estimator.ModeKeys.PREDICT:
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, features['input_seq'])

    readonly_component = image_embeddings

    # LSTM
    lstm_cell = LSTMRCell(num_units=params['num_lstm_units'], readonly_memory=readonly_component)

    # Dropout
    if mode == tf.estimator.ModeKeys.TRAIN:
        lstm_cell = contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=params['lstm_dropout_keep_prob'],
            output_keep_prob=params['lstm_dropout_keep_prob']
        )

    # VARS
    predict_res = None
    train_op = None
    total_loss = None
    eval_metric_ops = None

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = lstm_cell.zero_state(
            batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)

        # Allow the LSTM variables to be reused.
        lstm_scope.reuse_variables()

        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_res = predict_loop(lstm_cell=lstm_cell,
                                       embedding_map=embedding_map,
                                       initializer=initializer,
                                       initial_state=initial_state,
                                       features=features['features'])

        else:
            # dynamic_rnn
            sequence_length = tf.reduce_sum(labels['mask'], 1)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)

            # Stack batches vertically.
            lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    if mode != tf.estimator.ModeKeys.PREDICT:

        with tf.variable_scope("logits") as logits_scope:
            logits = contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=params['vocab_size'],
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope)
        # LOSS
        targets = tf.reshape(labels['target_seq'], [-1])
        weights = tf.to_float(tf.reshape(labels['mask'], [-1]))

        # Compute losses.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                                logits=logits)
        batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                            tf.reduce_sum(weights),
                            name="batch_loss")
        tf.losses.add_loss(batch_loss)
        total_loss = tf.losses.get_total_loss()
        if mode == tf.estimator.ModeKeys.EVAL:
            # EVAL
            with tf.variable_scope('perplexity',
                                   initializer=tf.constant_initializer(),
                                   dtype=tf.float32) as eval_scope:
                sum_losses = tf.get_variable('sum_losses', (), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
                sum_weights = tf.get_variable('sum_weights', (), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
                perplexity = tf.get_variable('perplexity', (), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

                sum_losses_update_op = tf.assign_add(sum_losses, tf.reduce_sum(losses * weights))
                sum_weights_update_op = tf.assign_add(sum_weights, tf.reduce_sum(weights))

                with tf.control_dependencies([sum_losses_update_op, sum_weights_update_op]):
                    perplexity_update_op = tf.assign(perplexity, tf.exp(sum_losses / sum_weights))
            eval_metric_ops = {
                'perplexity': (perplexity * 1.0, perplexity_update_op),
                'sum_weights': (sum_weights * 1.0, sum_weights_update_op),
                'sum_losses': (sum_losses * 1.0, sum_losses_update_op),
            }
        else:
            # TRAIN
            learning_rate = tf.constant(params['initial_learning_rate'])
            learning_rate_decay_fn = None
            if params['learning_rate_decay_factor'] > 0:
                num_batches_per_epoch = (params['num_examples_per_epoch'] /
                                         params['batch_size'])
                decay_steps = int(num_batches_per_epoch *
                                  params['num_epochs_per_decay'])

                def learning_rate_decay_fn(lr, global_step):
                    return tf.train.exponential_decay(
                        lr,
                        global_step,
                        decay_steps=decay_steps,
                        decay_rate=params['learning_rate_decay_factor'],
                        staircase=True)

            train_op = contrib.layers.optimize_loss(
                loss=total_loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate,
                optimizer=params['optimizer'],
                clip_gradients=params['clip_gradients'],
                learning_rate_decay_fn=learning_rate_decay_fn)

            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predict_res,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def predict_loop(lstm_cell, embedding_map, initializer, initial_state, params, features):
    beam_size = params['beam_size']
    initial_state = contrib.rnn.LSTMStateTuple(tf.tile(initial_state[0], [beam_size, 1]),
                                               tf.tile(initial_state[1], [beam_size, 1]))

    def body(time, pred_tuple, state_tuple):
        ides, coefs = pred_tuple
        with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, ides[-1])

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=seq_embeddings,
            state=state_tuple)

        # lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

        # TODO: Remove scope hacking by replacing contrib.layers.fully_connected with custom implantation
        def custom_getter(getter, name, *args, **kwargs):
            kwargs = kwargs.copy()
            kwargs['reuse'] = None
            name = name.replace('lstm/', '')
            return getter(name, *args, **kwargs)

        with tf.variable_scope("logits", custom_getter=custom_getter) as logits_scope:
            logits = contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=params['vocab_size'],
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope)

        prediction = tf.nn.softmax(logits, name="softmax")

        coef, w_i = tf.nn.top_k(prediction, k=beam_size)

        def repeat(x, n, shape):
            return tf.reshape(tf.tile(tf.reshape(x, [-1, 1]), [1, n]), shape)

        coefs = repeat(coefs, beam_size, [beam_size * beam_size])

        all_coefs = coefs + tf.log(tf.reshape(coef, [-1]))
        all_seq_i = tf.transpose(tf.concat([repeat(ides, beam_size, [-1, beam_size * beam_size]),
                                            tf.reshape(w_i, [1, -1])], axis=0))

        all_unique_coefs, idx = tf.unique(all_coefs)
        all_unique_seq_i = tf.gather(all_seq_i, tf.unique(idx)[0])

        unique_cond = tf.equal(time, 0)

        all_seq_i = tf.case([(unique_cond, lambda: all_unique_seq_i)],
                            default=lambda: all_seq_i)

        all_coefs = tf.case([(unique_cond, lambda: all_unique_coefs)],
                            default=lambda: all_coefs)

        top_cofs, top_id = tf.nn.top_k(all_coefs, k=beam_size)

        s_i = tf.gather(all_seq_i, top_id)
        with tf.control_dependencies([
            # tf.Print(ides, [time + 1, tf.transpose(s_i), top_cofs], summarize=1000)
        ]):
            return time + 1, (tf.transpose(s_i), top_cofs), state_tuple

    def cond(i, pred_tuple, _):
        ides, _ = pred_tuple
        cond_1 = i < params['seq_max_len']
        end_words = [params['end_word_index']] * beam_size
        cond_2 = tf.reduce_any(tf.not_equal(ides[i], end_words))
        with tf.control_dependencies([
            # tf.Print(ides, [ides[i], cond_2], summarize=1000)
        ]):
            return tf.logical_and(cond_1, cond_2)

    with tf.control_dependencies([tf.assert_equal(1, tf.shape(features)[0],
                                                  message='in PREDICT batch_size=1')]):
        i = tf.constant(0, dtype=tf.int32, name='i')
        initial_pred = (
            tf.constant([
                [params['start_word_index']] * beam_size
            ], dtype=tf.int32, name='word_indexes'),
            tf.constant([0.0] * beam_size, dtype=tf.float32, name='seq_coef'))

        seq_len, (ides, coefs), _ = tf.while_loop(cond=cond,
                                                  body=body,
                                                  loop_vars=[i, initial_pred, initial_state],
                                                  shape_invariants=[
                                                      i.shape,
                                                      (tf.TensorShape([None, beam_size]),
                                                       initial_pred[1].shape),
                                                      contrib.rnn.LSTMStateTuple(initial_state[0].shape,
                                                                                 initial_state[1].shape)
                                                  ],
                                                  parallel_iterations=1,
                                                  back_prop=False)

    with tf.control_dependencies([
        # tf.Print(ides, [ides, coefs], summarize=1000)
    ]):
        predict_res = {'ides': tf.expand_dims(ides, 0), 'coef': tf.expand_dims(coefs, 0)}
        return predict_res
