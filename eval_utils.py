import tensorflow as tf


def eval_perplexity(weights, losses):
    with tf.variable_scope('perplexity',
                           initializer=tf.constant_initializer(),
                           dtype=tf.float32):
        weights = tf.reshape(weights, [-1])
        sum_losses = tf.get_variable('sum_losses', (), trainable=False,
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])
        sum_weights = tf.get_variable('sum_weights', (), trainable=False,
                                      collections=[tf.GraphKeys.LOCAL_VARIABLES])
        perplexity = tf.get_variable('perplexity', (), trainable=False,
                                     collections=[tf.GraphKeys.LOCAL_VARIABLES])

        sum_losses_update_op = tf.assign_add(sum_losses, tf.reduce_sum(losses * weights))
        sum_weights_update_op = tf.assign_add(sum_weights, tf.reduce_sum(weights))

        with tf.control_dependencies([sum_losses_update_op, sum_weights_update_op]):
            perplexity_update_op = tf.assign(perplexity, tf.exp(sum_losses / sum_weights))
    return {
        'perplexity': (perplexity * 1.0, perplexity_update_op),
        'sum_weights': (sum_weights * 1.0, sum_weights_update_op),
        'sum_losses': (sum_losses * 1.0, sum_losses_update_op),
    }
