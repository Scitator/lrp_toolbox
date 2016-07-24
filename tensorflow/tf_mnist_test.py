# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# import input_data
mnist = input_data.read_data_sets("../data/MNIST", one_hot=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../python")

import render
import data_io

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
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
    Zs = tf.add(tf.expand_dims(tf.reduce_sum(Z, 1), 1),
                tf.expand_dims(tf.expand_dims(b, 0), 0))
    return tf.reduce_sum(
            (Z / Zs) * tf.expand_dims(R, 1),
            2)

# relevance propagation
def relevance_propagation(y_pred,
                          reversed_layers_inputs,
                          reversed_layers_weights,
                          reversed_layers_biases):
    assert len(reversed_layers_inputs) == len(reversed_layers_weights) == len(reversed_layers_biases)
    R = y_pred
    for i in range(len(reversed_layers_inputs)):
        R = _simple_lrp(R,
                        reversed_layers_inputs[i],
                        reversed_layers_weights[i],
                        reversed_layers_biases[i])
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
                              [layer_2_activations, layer_1_activations, x],
                              [weights['out'], weights['h2'], weights['h1']],
                              [biases['out'], biases['b2'], biases['b1']])

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

# I = np.random.permutation(mnist.test.images.shape[0])
I = range(12)

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    for inx in I[:12]:
        test_x = mnist.test.images[inx]
        test_y = mnist.test.labels[inx]
        relevance = sess.run(R,
                             feed_dict={
                                 x: test_x[np.newaxis, :]
                                 })
        pred_y = sess.run(pred,
                          feed_dict={
                              x: test_x[np.newaxis, :]
                              })

        digit = render.digit_to_rgb(test_x, scaling = 3)
        hm = render.hm_to_rgb(relevance, X = test_x, scaling = 3, sigma = 2)
        digit_hm = render.save_image([digit,hm],'./heatmap.png')
        data_io.write(relevance,'./heatmap.npy')

        print ('True Class:     {}'.format(np.argmax(test_y)))
        print ('Predicted Class: {}\n'.format(np.argmax(pred_y)))

        #display the image as written to file
        plt.imshow(digit_hm, interpolation = 'none', cmap=plt.cm.binary)
        plt.axis('off')
        plt.show()
