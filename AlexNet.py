import urllib.request
import math
import tensorflow as tf
import numpy as np
# myurl = 'https://s3.amazonaws.com/cse6250-nliu71/alldata1-50-50-20.npy'
# data = urllib.request.urlopen(myurl).read()
data = np.load("../100samples-112-112-96.npy") #load pre-processed data


IMG_SIZE_PX = 112
SLICE_COUNT = 96

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

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
    #Input image: 112*112*96
    ### 8 Layers:
    #       11 x 11 x 11 patches, 1 channel, 96 features to compute.
    #       5 x 5 x 5 patches, 96 channels, 256 features to compute.
    #       3 x 3 x 3 patches, 256 channels, 384 features to compute.
    #       3 x 3 x 3 patches, 384 channels, 384 features to compute.
    #       3 x 3 x 3 patches, 384 channels, 256 features to compute.
    weights = {'W_conv1':tf.Variable(tf.clip_by_value(tf.random_normal([11,11,11,1,96],stddev=1/math.sqrt(121*11)),-1,1)),
               'W_conv2':tf.Variable(tf.clip_by_value(tf.random_normal([5,5,5,96,256],stddev=1/math.sqrt(125*96)),-1,1)),
               'W_conv3':tf.Variable(tf.clip_by_value(tf.random_normal([3,3,3,256,384],stddev=1/math.sqrt(27*256)),-1,1)),
               'W_conv4':tf.Variable(tf.clip_by_value(tf.random_normal([3,3,3,384,384],stddev=1/math.sqrt(27*384)),-1,1)),
               'W_conv5':tf.Variable(tf.clip_by_value(tf.random_normal([3,3,3,384,256],stddev=1/math.sqrt(27*384)),-1,1)),
               'W_fc1':tf.Variable(tf.clip_by_value(tf.random_normal([602112,4096],stddev=1/math.sqrt(27)),-1,1)),
               'W_fc2':tf.Variable(tf.clip_by_value(tf.random_normal([4096,4096],stddev=1/math.sqrt(4096)),-1,1)),
               'out':tf.Variable(tf.clip_by_value(tf.random_normal([4096, n_classes],stddev=1/math.sqrt(4096)),-1,1))}

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

    ###conv-relu-pool * 2
    conv1 = conv3d(x, weights['W_conv1'],[1,4,4,4,1]) #11,11
    hidden1 = tf.nn.relu(conv1 + biases['b_conv1'])
    pool1 = maxpool3d(hidden1,[1,3,3,3,1],[1,2,2,2,1]) #overlapping pooling

    conv2 = conv3d(pool1, weights['W_conv2'],[1,1,1,1,1]) #5,5
    hidden2 = tf.nn.relu(conv2 + biases['b_conv2'])
    pool2 = maxpool3d(hidden2,[1,3,3,3,1],[1,2,2,2,1])  #F=3,S=2

    ###conv-relu * 3
    conv3 = tf.nn.relu(conv3d(pool2, weights['W_conv3'],[1,1,1,1,1]) + biases['b_conv3']) #3,3
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4'],[1,1,1,1,1]) + biases['b_conv4']) #3,3
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5'],[1,1,1,1,1]) + biases['b_conv5']) #3,3

    ###pool & normalize
    pool6 = maxpool3d(conv5,[1,3,3,3,1],[1,2,2,2,1]) #overlapping pooling
    pool6_normalized = tf.reshape(pool6, [-1, 602112])

    ###fc-relu-dropout * 2
    fc1 = tf.nn.relu(tf.matmul(pool6_normalized, weights['W_fc1'])+biases['b_fc1'])
    dropout1 = tf.nn.dropout(fc1, keep_rate)

    fc2 = tf.nn.relu(tf.matmul(dropout1, weights['W_fc2'])+biases['b_fc2'])
    dropout2 = tf.nn.dropout(fc2, keep_rate)

    output = tf.matmul(dropout2, weights['out'])+biases['out']
    return output

much_data = data
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-20]
validation_data = much_data[-20:]
import time

def train_neural_network(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate):
    prediction = AlexNet(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    gradients =   optimizer.compute_gradients(cost)
    capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gvs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    hm_epochs = 20
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())

        successful_runs = 0
        total_runs = 0

        for epoch in range(hm_epochs):
            start_time = time.time()
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([train_op, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    pass
                    #print(str(e))

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            print(epoch,"--- %s seconds ---" % (time.time() - start_time))
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)

train_neural_network(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)