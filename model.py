import tensorflow as tf


def im22txt(features, labels, mode, params):
    initializer = tf.random_uniform_initializer(
        minval=-params['initializer_scale'],
        maxval=params['initializer_scale'])
    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=features['features'],
            num_outputs=params['embedding_size'],
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=None,
            scope=scope)

    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name="map",
            shape=[params['vocab_size'], params['embedding_size']],
            initializer=initializer)
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, features['input_seq'])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=params['num_lstm_units'], state_is_tuple=True)
    if mode == tf.estimator.ModeKeys.TRAIN:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=params['lstm_dropout_keep_prob'],
            output_keep_prob=params['lstm_dropout_keep_prob']
        )

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = lstm_cell.zero_state(
            batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)

        # Allow the LSTM variables to be reused.
        lstm_scope.reuse_variables()
        lstm_outputs = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            # TODO


            # In inference mode, use concatenated states for convenient feeding and
            # fetching.
            tf.concat(axis=1, values=initial_state, name="initial_state")

            # Placeholder for feeding a batch of concatenated states.
            state_feed = tf.placeholder(dtype=tf.float32,
                                        shape=[None, sum(lstm_cell.state_size)],
                                        name="state_feed")
            state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

            # Run a single LSTM step.
            lstm_outputs, state_tuple = lstm_cell(
                inputs=tf.squeeze(seq_embeddings, axis=[1]),
                state=state_tuple)

            # Concatentate the resulting state.
            tf.concat(axis=1, values=state_tuple, name="state")
        else:
            # Run the batch of sequence embeddings through the LSTM.
            sequence_length = tf.reduce_sum(labels['mask'], 1)
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
        logits = tf.contrib.layers.fully_connected(
            inputs=lstm_outputs,
            num_outputs=params['vocab_size'],
            activation_fn=None,
            weights_initializer=initializer,
            scope=logits_scope)

    prediction = None
    train_op = None
    total_loss = None
    eval_metric_ops = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        prediction = tf.nn.softmax(logits, name="softmax")
    else:
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

        with tf.variable_scope('perplexity', initializer=tf.constant_initializer()) as eval_scope:
            sum_losses = tf.get_variable('sum_losses', (), trainable=False, dtype=tf.float32)
            sum_weights = tf.get_variable('sum_weights', (), trainable=False, dtype=tf.float32)
            perplexity = tf.get_variable('perplexity', (), trainable=False, dtype=tf.float32)

            sum_losses_update_op = tf.assign_add(sum_losses, tf.reduce_sum(losses * weights))
            sum_weights_update_op = tf.assign_add(sum_weights, tf.reduce_sum(weights))

            with tf.control_dependencies([sum_losses_update_op, sum_weights_update_op]):
                perplexity_update_op = tf.assign(perplexity, tf.exp(sum_losses / sum_weights))

        eval_metric_ops = {
            'perplexity': (perplexity * 1.0, perplexity_update_op),
            'sum_weights': (sum_weights * 1.0, sum_weights_update_op),
            'sum_losses': (sum_losses * 1.0, sum_losses_update_op),
        }

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

        train_op = tf.contrib.layers.optimize_loss(
            loss=total_loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            optimizer=params['optimizer'],
            clip_gradients=params['clip_gradients'],
            learning_rate_decay_fn=learning_rate_decay_fn)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=prediction,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
