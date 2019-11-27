import tensorflow as tf


def mlp(x, keep_prob, is_train, conf):

    def initializers(bias_init=conf.bias_init, beta=conf.l2_str):
        weight_init = tf.initializers.variance_scaling(scale=2.0,
                                                       # He initialization https://arxiv.org/pdf/1502.01852v1.pdf
                                                       mode='fan_in',
                                                       distribution='normal'
                                                       )
        bias_ini = tf.constant_initializer(value=bias_init)  # small constant
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
        return weight_init, bias_ini, regularizer

    def new_fc_layer(inputs,  # The previous layer.
                     num_outputs,  # Num. outputs.
                     name):  # Name
        with tf.variable_scope(name):
            w_ini, b_ini, reg_ini = initializers()
            layer = tf.layers.dense(inputs=inputs,
                                    units=num_outputs,
                                    kernel_initializer=w_ini,
                                    bias_initializer=b_ini,
                                    kernel_regularizer=reg_ini,
                                    use_bias=True,
                                    activation=tf.nn.relu,
                                    name='FC',
                                    )
            return layer

    w_ini, b_ini, reg_ini = initializers()

    # x_multichannel = tf.reshape(x, [-1, conf.seq_lngth, conf.num_ch])
    # net = tf.layers.flatten(x_multichannel, 'flat_layer')
    net = new_fc_layer(x, 128, "fc1_relu_bn")
    # net = new_fc_layer(net, 128, "fc1_relu_bn2")

    # apply dropout to first FC layer
    net = tf.nn.dropout(net, keep_prob, name='Dropout')
    # 2nd fully connected layer = output
    logits = tf.layers.dense(inputs=net, units=conf.num_cl, activation=None, kernel_initializer=w_ini,
                             bias_initializer=b_ini, kernel_regularizer=reg_ini, name='Output')
    return logits
