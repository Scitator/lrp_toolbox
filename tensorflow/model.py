# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# import input_data
mnist = input_data.read_data_sets("../data/MNIST", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.005
training_epochs = 15
batch_size = 16
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

R = tf.Variable("float", [None, n_input])

def _simple_lrp(R, X, W, b):
    Z = tf.expand_dims(W, 0) * tf.expand_dims(X, -1)
    Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + \
                tf.expand_dims(tf.expand_dims(b, 0), 0)
    return tf.reduce_sum(
            (Z / Zs) * tf.expand_dims(R, 1),
            2)

# relevance propagation
def relevance_propagation(y_pred,
                          reversed_layers):
    # softmax lrp
    R = tf.nn.softmax(y_pred) * y_pred
    for l_input, l_weight, l_bias in reversed_layers:
        R = _simple_lrp(R,
                        l_input,
                        l_weight,
                        l_bias)
    return R

# Create model
def multilayer_perceptron_lrp(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1_activations = tf.nn.tanh(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1_activations, weights['h2']), biases['b2'])
    layer_2_activations = tf.nn.tanh(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2_activations, weights['out']) + biases['out']

    R = relevance_propagation(out_layer,
                              zip(
                                  [layer_2_activations, layer_1_activations, x],
                                  [weights['out'], weights['h2'], weights['h1']],
                                  [biases['out'], biases['b2'], biases['b1']]))

    return out_layer, R

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred, R = multilayer_perceptron_lrp(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()
model_path = "model.ckpt"
