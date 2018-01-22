import tensorflow as tf
import tensorflow.contrib as contrib


def feature2seq(features,
                input_seq,
                mask,
                mode,
                vocab_size,
                predictor,
                initializer_scale,
                embedding_size,
                num_lstm_units,
                lstm_dropout_keep_prob,
                features_dropout_keep_prob):
    initializer = tf.random_uniform_initializer(
        minval=-initializer_scale,
        maxval=initializer_scale
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        features = tf.layers.dropout(features, 1.0 - features_dropout_keep_prob, training=True)

    # Image Embedding
    with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=features,
            num_outputs=embedding_size,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=tf.zeros_initializer,
            scope=scope)

    # Seq Embedding
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
        embedding_map = tf.get_variable(
            name="map",
            shape=[vocab_size, embedding_size],
            initializer=initializer)
        if mode != tf.estimator.ModeKeys.PREDICT:
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seq)
        else:
            seq_embeddings = None

    # LSTM
    lstm_cell = contrib.rnn.BasicLSTMCell(
        num_units=num_lstm_units, state_is_tuple=True)

    # Dropout
    if mode == tf.estimator.ModeKeys.TRAIN:
        lstm_cell = contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=lstm_dropout_keep_prob,
            output_keep_prob=lstm_dropout_keep_prob
        )

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
        # Feed the image embeddings to set the initial LSTM state.
        zero_state = lstm_cell.zero_state(
            batch_size=image_embeddings.get_shape()[0], dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)

        # Allow the LSTM variables to be reused.
        lstm_scope.reuse_variables()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return predictor(
                initial_state=initial_state,
                lstm_cell=lstm_cell,
                embedding_map=embedding_map
            )
        else:
            # dynamic_rnn
            sequence_length = tf.reduce_sum(mask, 1, name='sequence_length')
            lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                inputs=seq_embeddings,
                                                sequence_length=sequence_length,
                                                initial_state=initial_state,
                                                dtype=tf.float32,
                                                scope=lstm_scope)

            # Stack batches vertically.
            lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size], name='lstm_outputs')

    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("logits") as logits_scope:
            logits = contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=vocab_size,
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope)

        return {'logits': logits}
