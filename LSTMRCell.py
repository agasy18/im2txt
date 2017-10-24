import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.python.ops.rnn_cell_impl import _linear, LSTMStateTuple


class LSTMRCell(contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, readonly_memory, forget_bias=1.0, activation=None, reuse=None):
        super(LSTMRCell, self).__init__(
            num_units=num_units,
            forget_bias=forget_bias,
            state_is_tuple=True,
            activation=activation,
            reuse=reuse)
        self.readonly_memory = readonly_memory

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell with read only unit (LSTM).

            Args:
              inputs: `2-D` tensor with shape `[batch_size x input_size]`.
              state: An `LSTMStateTuple` of state tensors, each shaped
                `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
                `True`.  Otherwise, a `Tensor` shaped
                `[batch_size x 2 * self.state_size]`.

            Returns:
              A pair containing the new hidden state, and the new state (either a
                `LSTMStateTuple` or a concatenated state, depending on
                `state_is_tuple`).
            """
        sigmoid = tf.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        concat = _linear([inputs, h], 5 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate, r = readonly_gate
        i, j, f, o, r = tf.split(value=concat, num_or_size_splits=5, axis=1)

        new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j) + sigmoid(r) * sigmoid(self.readonly_memory)
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)
        return new_h, new_state
