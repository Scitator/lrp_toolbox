from model import *

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../python")

import render
import data_io

# I = np.random.permutation(mnist.test.images.shape[0])
I = range(12)

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    for inx in I[:12]:
        test_x = mnist.test.images[inx]
        test_x = (test_x - 0.5) * 2
        test_y = mnist.test.labels[inx]
        relevance = sess.run(R,
                             feed_dict={
                                 x: test_x[np.newaxis, :]
                                 })
        # import pdb; pdb.set_trace()
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
