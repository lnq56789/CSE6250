# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 21:20:23 2017

@author: Zimu
"""

#import dicom # for reading dicom files
import os # for doing directory operations
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import numpy as np
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image as img
from skimage import measure#, morphology
#from skimage.segmentation import slic
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import tensorflow as tf
from six.moves import xrange


EVAL_BATCH_SIZE = 1
IMG_SIZE_PX = 64
SLICE_COUNT = 24
n_classes = 2
processData = False
keep_rate = 0.8

def conv3d(x, W, s):
    return tf.nn.conv3d(x, W, strides=s, padding='SAME')

def maxpool3d(x,k,s):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=k, strides=s, padding='SAME')

def ClipIfNotNone(grad):
#    Credit to azni at http://stackoverflow.com/questions/39295136/gradient-clipping-appears-to-choke-on-none for solving none gradient problem
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)
  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.

def AlexNet(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate):
    #Input image: 64*64*24
    ### 8 Layers:
    #       11 x 11 x 11 patches, 1 channel, 96 features to compute.
    #       5 x 5 x 5 patches, 96 channels, 256 features to compute.
    #       3 x 3 x 3 patches, 256 channels, 384 features to compute.
    #       3 x 3 x 3 patches, 384 channels, 384 features to compute.
    #       3 x 3 x 3 patches, 384 channels, 256 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([11,11,11,1,96],stddev=math.sqrt(2/1331))),
               'W_conv2':tf.Variable(tf.random_normal([5,5,5,96,256],stddev=math.sqrt(2/12000))),
               'W_conv3':tf.Variable(tf.random_normal([3,3,3,256,384],stddev=math.sqrt(2/6912))),
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,384,384],stddev=math.sqrt(2/10368))),
               'W_conv5':tf.Variable(tf.random_normal([3,3,3,384,256],stddev=math.sqrt(2/10368))),
               'W_fc1':tf.Variable(tf.random_normal([1024,4096],stddev=math.sqrt(2/1024))),
               'W_fc2':tf.Variable(tf.random_normal([4096,4096],stddev=math.sqrt(2/4096))),
               'out':tf.Variable(tf.random_normal([4096, n_classes],stddev=math.sqrt(2/4096)))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([96])),
              'b_conv2':tf.Variable(tf.random_normal([256])),
              'b_conv3':tf.Variable(tf.random_normal([384])),
              'b_conv4':tf.Variable(tf.random_normal([384])),
              'b_conv5':tf.Variable(tf.random_normal([256])),
              'b_fc1':tf.Variable(tf.random_normal([4096])),
              'b_fc2':tf.Variable(tf.random_normal([4096])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    #normalize                      image X     image Y     image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    ###conv-relu-pool-norm * 2
    conv1 = conv3d(x, weights['W_conv1'],[1,4,4,4,1]) #11,11
    hidden1 = tf.nn.relu(conv1 + biases['b_conv1'])
    pool1 = maxpool3d(hidden1,[1,3,3,3,1],[1,2,2,2,1]) #overlapping pooling

    conv2 = conv3d(pool1, weights['W_conv2'],[1,1,1,1,1]) #5,5
    hidden2 = tf.nn.relu(conv2 + biases['b_conv2'])
    pool2 = maxpool3d(hidden2,[1,3,3,3,1],[1,2,2,2,1])

    ###conv * 3
    conv3 = tf.nn.relu(conv3d(pool2, weights['W_conv3'],[1,1,1,1,1]) + biases['b_conv3'])
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4'],[1,1,1,1,1]) + biases['b_conv4'])
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5'],[1,1,1,1,1]) + biases['b_conv5'])

    ###pool & normalize
    pool6 = maxpool3d(conv5,[1,3,3,3,1],[1,2,2,2,1]) #overlapping pooling
    pool6_normalized = tf.reshape(pool6, [-1, 1024])

    ###fc-relu-dropout * 2
    fc1 = tf.nn.relu(tf.matmul(pool6_normalized, weights['W_fc1'])+biases['b_fc1'])
    dropout1 = tf.nn.dropout(fc1, keep_rate)

    fc2 = tf.nn.relu(tf.matmul(dropout1, weights['W_fc2'])+biases['b_fc2'])
    dropout2 = tf.nn.dropout(fc2, keep_rate)

    output = tf.matmul(dropout2, weights['out'])+biases['out']
    return output

def eval_in_batches(data, sess,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate):

    x = tf.placeholder('float')
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, n_classes), dtype=np.float32)
    for begin in xrange(0, size):
        inputX = data[begin]
        eval_prediction = tf.nn.softmax(UConvNet(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate))
        sess.run(tf.global_variables_initializer())
        predictions[begin] = sess.run(eval_prediction,feed_dict={x: inputX})
    return predictions
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  accurate = 0
  for index in range(0,predictions.shape[0]):
      if np.argmax(predictions[index]) == np.argmax(labels[index]):
          accurate = accurate + 1
  return(1-accurate/predictions.shape[0])


def train_alex_net(x,y,train_data,validation_data,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate):
    import time
    prediction = AlexNet(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
#    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#   apply gradient clipping to eliminate gradient explosion problem
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            start = time.time()
            print("Start epoch {0}".format(epoch+1))
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    print(str(e))
            end = time.time()
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print('Epoch', epoch+1, '--- %s seconds ---' % (end - start))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)

def main():

    import time

#    At this point, we've got the list of patients by their IDs, and their associated labels stored in a dataframe.
#    Now, we can begin to iterate through the patients and gather their respective data.
#    We're almost certainly going to need to do some preprocessing of this data, but we'll see.

    x = tf.placeholder('float')
    y = tf.placeholder('float')

    much_data1 = np.load('alldata4-224-224-64.npy')

    train_data1 = much_data1[:-300]
    validation_data1 = much_data1[-300:]

    start_time = time.time()
#    train_neural_network(x,y,train_data1,validation_data1,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
    train_alex_net(x,y,train_data1,validation_data1,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
    end_time = time.time()
    print("Total process for all 224*224*64 images takes %s seconds." % (end_time - start_time))

if __name__ == "__main__":

	main()
