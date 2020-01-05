import tensorflow as tf

class Core_layers(object):

    @staticmethod
    def multihead_graph_attention_v1(node_features,
                        node_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):


        
        node_features = tf.expand_dims(node_features,0)
        node_keys     = node_features


        with tf.variable_scope(scope, reuse=reuse):
            if node_units is None:  # set default size for attention size C
                node_units = node_features.get_shape().as_list()[-1]

            # Linear Projections
            Q = tf.layers.dense(node_features, node_units, activation=tf.nn.relu)  # [N, T_q, C]
            K = tf.layers.dense(node_keys, node_units, activation=tf.nn.relu)  # [N, T_k, C]
            V = tf.layers.dense(node_keys, node_units, activation=tf.nn.relu)  # [N, T_k, C]

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
            K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
            V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]

            # Attention
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)

            # Scale : outputs = outputs / sqrt( d_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            # see : https://github.com/Kyubyong/transformer/issues/3
            key_masks = tf.sign(tf.abs(tf.reduce_sum(node_keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(node_features)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # -infinity
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation: outputs is a weight matrix
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(node_features, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(node_keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # reshape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            return outputs

    @staticmethod
    def add_and_normalize(node_features, attention_output):
        # residual connection
        attention_output += node_features

        # layer normaliztion
        outputs = Core_layers.layer_normalization(attention_output)
        outputs = tf.squeeze(outputs)
        return outputs

    @staticmethod
    def layer_normalization(inputs,
                        epsilon=1e-8,
                        scope="ln",
                        reuse=None):
        
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)
            outputs = gamma * normalized + beta

        return outputs

    @staticmethod
    def feed_forward(attention_input,node_units, act = tf.nn.relu, bias = True):

        
        # dense layer with xavier weights
        fc_layer = tf.get_variable(name='feed_forward',
                                   shape=[attention_input.shape[-1],node_units],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer())
        
        if bias:
            # bias 
            bias    = tf.get_variable(name='bias',
                                    shape=[node_units],
                                    dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer())
        
            #final output 
            output = act(tf.add(tf.matmul(attention_input,fc_layer),bias))
        else:
            output = act(tf.matmul(attention_input,fc_layer))

        return output

    

    @staticmethod
    def InnerProductDecoder(latent_input,
                            input_dim, 
                            dropout=0., 
                            act=tf.nn.sigmoid):

        inputs = tf.nn.dropout(latent_input, 1- dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = act(x)
        return outputs

