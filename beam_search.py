import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple


def beam_search(initial_state,
                lstm_cell,
                embedding_map,
                beam_size,
                vocab_size,
                start_word_index,
                end_word_index,
                seq_max_len):
    initial_state = LSTMStateTuple(tf.tile(initial_state[0], [beam_size, 1]),
                                   tf.tile(initial_state[1], [beam_size, 1]))

    # while
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
                num_outputs=vocab_size,
                activation_fn=None,
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
        cond_1 = i < seq_max_len
        end_words = [end_word_index] * beam_size
        cond_2 = tf.reduce_any(tf.not_equal(ides[i], end_words))
        with tf.control_dependencies([
            # tf.Print(ides, [ides[i], cond_2], summarize=1000)
        ]):
            return tf.logical_and(cond_1, cond_2)

    with tf.control_dependencies([tf.assert_equal(1, tf.shape(initial_state)[0],
                                                  message='BEAM Search batch_size=1')]):
        i = tf.constant(0, dtype=tf.int32, name='i')
        initial_pred = (
            tf.constant([
                [start_word_index] * beam_size
            ], dtype=tf.int32, name='word_indexes'),
            tf.constant([0.0] * beam_size, dtype=tf.float32, name='seq_coef'))

        seq_len, (ides, coefs), _ = tf.while_loop(cond=cond,
                                                  body=body,
                                                  loop_vars=[i, initial_pred, initial_state],
                                                  shape_invariants=[
                                                      i.shape,
                                                      (tf.TensorShape([None, beam_size]),
                                                       initial_pred[1].shape),
                                                      LSTMStateTuple(initial_state[0].shape,
                                                                     initial_state[1].shape)
                                                  ],
                                                  parallel_iterations=1,
                                                  back_prop=False)

    with tf.control_dependencies([
        # tf.Print(ides, [ides, coefs], summarize=1000)
    ]):
        ides = tf.expand_dims(ides, 0, name='ides')
        coefs = tf.expand_dims(coefs, 0, name='coefs')
        return ides, coefs
