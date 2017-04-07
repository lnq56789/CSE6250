import numpy as np
data = np.load('../100sample-224-224-24.npy') #load pre-processed data

import tensorflow as tf
import numpy as np

IMG_SIZE_PX = 224
SLICE_COUNT = 24

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.6

def conv3d(x, W, s):
    return tf.nn.conv3d(x, W, strides=s, padding='SAME')

def maxpool3d(x,k,s):
    #                        size of window    movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=k, strides=s, padding='SAME')

def avgpool3d(x,k,s):
    #                        size of window    movement of window as you slide about
    return tf.nn.avg_pool3d(x, ksize=k, strides=s, padding='VALID')

def depthconcat(inputs):
    concat_dim = 3
    shapes = []
    for input_ in inputs:
        shapes.append(tf.to_float(tf.shape(input_)[:3]))
    shape_tensor = tf.pack(shapes)
    max_dims = tf.reduce_max(shape_tensor, 0)

    padded_inputs = []
    for idx, input_ in enumerate(inputs):
        mean_diff = (max_dims - shapes[idx])/2.0
        pad_low = tf.floor(mean_diff)
        pad_high = tf.ceil(mean_diff)
        paddings = tf.to_int32(tf.pack([pad_low, pad_high], axis=1))
        paddings = tf.pad(paddings, paddings=[[0, 1], [0, 0]])
        padded_inputs.append(tf.pad(input_, paddings))

    return tf.concat(concat_dim, padded_inputs, name=name)

def google_net(x):
    # input shape(224*224*)
    ### 22 Layers:
                # conv1: 7 x 7 x 7 patches, 1 depth(channel), 64 filters
    weights = {'W_conv1':tf.Variable(tf.random_normal([7,7,7,1,64],stddev=math.sqrt(2/7*7*7*1))), # need to set stddev=math.sqrt(2/units) according to https://arxiv.org/abs/1502.01852
                # conv2_reduce: 1 x 1 x 1 patches, 64 depth, 64 filters
               'W_conv2_reduce':tf.Variable(tf.random_normal([1,1,1,64,64],stddev=math.sqrt(2/64))),
                # conv2: 3 x 3 x 3 patches, 64 depth, 192 filters
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,64,192],stddev=math.sqrt(2/3*3*3*64))),
                # inception(3a)
               'W_inception_3a_1x1':tf.Variable(tf.random_normal([1,1,1,192,64],stddev=math.sqrt(2/192))),
               'W_inception_3a_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,192,96],stddev=math.sqrt(2/64))),
               'W_inception_3a_3x3':tf.Variable(tf.random_normal([3,3,3,96,128])),
               'W_inception_3a_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,192,16])),
               'W_inception_3a_5x5':tf.Variable(tf.random_normal([5,5,5,16,32])),
               'W_inception_3a_pool_proj':tf.Variable(tf.random_normal([1,1,1,192,32])),

               'W_inception_3b_1x1':tf.Variable(tf.random_normal([1,1,1,256,128])),
               'W_inception_3b_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,256,128])),
               'W_inception_3b_3x3':tf.Variable(tf.random_normal([3,3,3,128,192])),
               'W_inception_3b_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,256,32])),
               'W_inception_3b_5x5':tf.Variable(tf.random_normal([5,5,5,32,96])),
               'W_inception_3b_pool_proj':tf.Variable(tf.random_normal([1,1,1,256,64])),

               'W_inception_4a_1x1':tf.Variable(tf.random_normal([1,1,1,480,192])),
               'W_inception_4a_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,480,96])),
               'W_inception_4a_3x3':tf.Variable(tf.random_normal([3,3,3,96,208])),
               'W_inception_4a_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,480,16])),
               'W_inception_4a_5x5':tf.Variable(tf.random_normal([5,5,5,16,48])),
               'W_inception_4a_pool_proj':tf.Variable(tf.random_normal([1,1,1,480,64])),

               'W_inception_4b_1x1':tf.Variable(tf.random_normal([1,1,1,512,160])),
               'W_inception_4b_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,512,112])),
               'W_inception_4b_3x3':tf.Variable(tf.random_normal([3,3,3,112,224])),
               'W_inception_4b_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,512,24])),
               'W_inception_4b_5x5':tf.Variable(tf.random_normal([5,5,5,24,64])),
               'W_inception_4b_pool_proj':tf.Variable(tf.random_normal([1,1,1,512,64])),

               'W_inception_4c_1x1':tf.Variable(tf.random_normal([1,1,1,512,128])),
               'W_inception_4c_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,512,128])),
               'W_inception_4c_3x3':tf.Variable(tf.random_normal([3,3,3,128,256])),
               'W_inception_4c_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,512,24])),
               'W_inception_4c_5x5':tf.Variable(tf.random_normal([5,5,5,24,64])),
               'W_inception_4c_pool_proj':tf.Variable(tf.random_normal([1,1,1,512,64])),

               'W_inception_4d_1x1':tf.Variable(tf.random_normal([1,1,1,512,112])),
               'W_inception_4d_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,512,144])),
               'W_inception_4d_3x3':tf.Variable(tf.random_normal([3,3,3,144,288])),
               'W_inception_4d_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,512,32])),
               'W_inception_4d_5x5':tf.Variable(tf.random_normal([5,5,5,32,64])),
               'W_inception_4d_pool_proj':tf.Variable(tf.random_normal([1,1,1,512,64])),

               'W_inception_4e_1x1':tf.Variable(tf.random_normal([1,1,1,528,256])),
               'W_inception_4e_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,528,160])),
               'W_inception_4e_3x3':tf.Variable(tf.random_normal([3,3,3,160,320])),
               'W_inception_4e_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,528,32])),
               'W_inception_4e_5x5':tf.Variable(tf.random_normal([5,5,5,32,128])),
               'W_inception_4e_pool_proj':tf.Variable(tf.random_normal([1,1,1,528,128])),

               'W_inception_5a_1x1':tf.Variable(tf.random_normal([1,1,1,832,256])),
               'W_inception_5a_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,832,160])),
               'W_inception_5a_3x3':tf.Variable(tf.random_normal([3,3,3,160,320])),
               'W_inception_5a_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,832,32])),
               'W_inception_5a_5x5':tf.Variable(tf.random_normal([5,5,5,32,128])),
               'W_inception_5a_pool_proj':tf.Variable(tf.random_normal([1,1,1,832,128])),

               'W_inception_5b_1x1':tf.Variable(tf.random_normal([1,1,1,832,384])),
               'W_inception_5b_3x3_reduce':tf.Variable(tf.random_normal([1,1,1,832,192])),
               'W_inception_5b_3x3':tf.Variable(tf.random_normal([3,3,3,192,384])),
               'W_inception_5b_5x5_reduce':tf.Variable(tf.random_normal([1,1,1,832,48])),
               'W_inception_5b_5x5':tf.Variable(tf.random_normal([5,5,5,48,128])),
               'W_inception_5b_pool_proj':tf.Variable(tf.random_normal([1,1,1,832,128])),

               'W_fc':tf.Variable(tf.random_normal([49*49*256,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
              'b_conv2_reduce':tf.Variable(tf.random_normal([64])),
              'b_conv2':tf.Variable(tf.random_normal([192])),

              'b_inception_3a_1x1':tf.Variable(tf.random_normal([64])),
              'b_inception_3a_3x3_reduce':tf.Variable(tf.random_normal([96])),
              'b_inception_3a_3x3':tf.Variable(tf.random_normal([128])),
              'b_inception_3a_5x5_reduce':tf.Variable(tf.random_normal([16])),
              'b_inception_3a_5x5':tf.Variable(tf.random_normal([32])),
              'b_inception_3a_pool_proj':tf.Variable(tf.random_normal([32])),

              'b_inception_3b_1x1':tf.Variable(tf.random_normal([128])),
              'b_inception_3b_3x3_reduce':tf.Variable(tf.random_normal([128])),
              'b_inception_3b_3x3':tf.Variable(tf.random_normal([192])),
              'b_inception_3b_5x5_reduce':tf.Variable(tf.random_normal([32])),
              'b_inception_3b_5x5':tf.Variable(tf.random_normal([96])),
              'b_inception_3b_pool_proj':tf.Variable(tf.random_normal([64])),

              'b_inception_4a_1x1':tf.Variable(tf.random_normal([192])),
              'b_inception_4a_3x3_reduce':tf.Variable(tf.random_normal([96])),
              'b_inception_4a_3x3':tf.Variable(tf.random_normal([208])),
              'b_inception_4a_5x5_reduce':tf.Variable(tf.random_normal([16])),
              'b_inception_4a_5x5':tf.Variable(tf.random_normal([48])),
              'b_inception_4a_pool_proj':tf.Variable(tf.random_normal([64])),

              'b_inception_4b_1x1':tf.Variable(tf.random_normal([160])),
              'b_inception_4b_3x3_reduce':tf.Variable(tf.random_normal([112])),
              'b_inception_4b_3x3':tf.Variable(tf.random_normal([224])),
              'b_inception_4b_5x5_reduce':tf.Variable(tf.random_normal([24])),
              'b_inception_4b_5x5':tf.Variable(tf.random_normal([64])),
              'b_inception_4b_pool_proj':tf.Variable(tf.random_normal([64])),

              'b_inception_4c_1x1':tf.Variable(tf.random_normal([128])),
              'b_inception_4c_3x3_reduce':tf.Variable(tf.random_normal([128])),
              'b_inception_4c_3x3':tf.Variable(tf.random_normal([256])),
              'b_inception_4c_5x5_reduce':tf.Variable(tf.random_normal([24])),
              'b_inception_4c_5x5':tf.Variable(tf.random_normal([64])),
              'b_inception_4c_pool_proj':tf.Variable(tf.random_normal([64])),

              'b_inception_4d_1x1':tf.Variable(tf.random_normal([112])),
              'b_inception_4d_3x3_reduce':tf.Variable(tf.random_normal([144])),
              'b_inception_4d_3x3':tf.Variable(tf.random_normal([288])),
              'b_inception_4d_5x5_reduce':tf.Variable(tf.random_normal([32])),
              'b_inception_4d_5x5':tf.Variable(tf.random_normal([64])),
              'b_inception_4d_pool_proj':tf.Variable(tf.random_normal([64])),

              'b_inception_4e_1x1':tf.Variable(tf.random_normal([256])),
              'b_inception_4e_3x3_reduce':tf.Variable(tf.random_normal([160])),
              'b_inception_4e_3x3':tf.Variable(tf.random_normal([320])),
              'b_inception_4e_5x5_reduce':tf.Variable(tf.random_normal([32])),
              'b_inception_4e_5x5':tf.Variable(tf.random_normal([128])),
              'b_inception_4e_pool_proj':tf.Variable(tf.random_normal([128])),

              'b_inception_5a_1x1':tf.Variable(tf.random_normal([256])),
              'b_inception_5a_3x3_reduce':tf.Variable(tf.random_normal([160])),
              'b_inception_5a_3x3':tf.Variable(tf.random_normal([320])),
              'b_inception_5a_5x5_reduce':tf.Variable(tf.random_normal([32])),
              'b_inception_5a_5x5':tf.Variable(tf.random_normal([128])),
              'b_inception_5a_pool_proj':tf.Variable(tf.random_normal([128])),

              'b_inception_5b_1x1':tf.Variable(tf.random_normal([384])),
              'b_inception_5b_3x3_reduce':tf.Variable(tf.random_normal([192])),
              'b_inception_5b_3x3':tf.Variable(tf.random_normal([384])),
              'b_inception_5b_5x5_reduce':tf.Variable(tf.random_normal([48])),
              'b_inception_5b_5x5':tf.Variable(tf.random_normal([128])),
              'b_inception_5b_pool_proj':tf.Variable(tf.random_normal([128])),

              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    # normalize                      image X     image Y     image Z
    x = tf.reshape(x, shape=[-1, IMG_SIZE_PX, IMG_SIZE_PX, SLICE_COUNT, 1])

    ### conv-pool-lrn-conv-conv-lrn-pool
    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1'],[1,2,2,2,1])+biases['b_conv1'])
    pool1 = maxpool3d(conv1,[1,3,3,3,1],[1,2,2,2,1])
    # pool1_normalized = tf.reshape(pool1, [-1, 6*6*3*256])
    lrn1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=2e-05,beta=0.75)

    conv2_reduce = tf.nn.relu(conv3d(lrn1, weights['W_conv2_reduce'],[1,1,1,1,1])+ biases['b_conv2_reduce'])
    conv2 = tf.nn.relu(conv3d(conv2_reduce, weights['W_conv2'],[1,1,1,1,1])+ biases['b_conv2'])
    lrn2 = tf.nn.lrn(conv2,5,bias=1.0,alpha=2e-05,beta=0.75)
    pool2 = maxpool3d(lrn2,[1,3,3,3,1],[1,2,2,2,1])

    ### inception(3a)
    inception_3a_1x1 = tf.nn.relu(conv3d(pool2, weights['W_inception_3a_1x1'],[1,1,1,1,1])+biases['b_inception_3a_1x1'])

    inception_3a_3x3_reduce = tf.nn.relu(conv3d(pool2, weights['W_inception_3a_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_3a_3x3_reduce'])
    inception_3a_3x3 = tf.nn.relu(conv3d(inception_3a_3x3_reduce, weights['W_inception_3a_3x3'],[1,1,1,1,1])+biases['b_inception_3a_3x3'])

    inception_3a_5x5_reduce = tf.nn.relu(conv3d(pool2, weights['W_inception_3a_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_3a_5x5_reduce'])
    inception_3a_5x5 = tf.nn.relu(conv3d(inception_3a_5x5_reduce, weights['W_inception_3a_5x5'],[1,1,1,1,1])+biases['b_inception_3a_5x5'])

    inception_3a_pool = maxpool3d(pool2,[1,3,3,3,1],[1,2,2,2,1])
    inception_3a_pool_proj = tf.nn.relu(conv3d(inception_3a_pool, weights['W_inception_3a_pool_proj'],[1,1,1,1,1])+biases['b_inception_3a_pool_proj'])

    inception_3a_output = depthconcat([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj])


    ### inception(3b)
    inception_3b_1x1 = tf.nn.relu(conv3d(inception_3a_output, weights['W_inception_3b_1x1'],[1,1,1,1,1])+biases['b_inception_3b_1x1'])

    inception_3b_3x3_reduce = tf.nn.relu(conv3d(inception_3a_output, weights['W_inception_3b_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_3b_3x3_reduce'])
    inception_3b_3x3 = tf.nn.relu(conv3d(inception_3b_3x3_reduce, weights['W_inception_3b_3x3'],[1,1,1,1,1])+biases['b_inception_3b_3x3'])

    inception_3b_5x5_reduce = tf.nn.relu(conv3d(inception_3a_output, weights['W_inception_3b_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_3b_5x5_reduce'])
    inception_3b_5x5 = tf.nn.relu(conv3d(inception_3b_5x5_reduce, weights['W_inception_3b_5x5'],[1,1,1,1,1])+biases['b_inception_3b_5x5'])

    inception_3b_pool = maxpool3d(inception_3a_output,[1,3,3,3,1],[1,2,2,2,1])
    inception_3b_pool_proj = tf.nn.relu(conv3d(inception_3b_pool, weights['W_inception_3b_pool_proj'],[1,1,1,1,1])+biases['b_inception_3b_pool_proj'])

    inception_3b_output = depthconcat([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj])
    pool3 = maxpool3d(inception_3b_output,[1,3,3,3,1],[1,2,2,2,1])

    ### inception(4a)
    inception_4a_1x1 = tf.nn.relu(conv3d(pool3, weights['W_inception_4a_1x1'],[1,1,1,1,1])+biases['b_inception_4a_1x1'])

    inception_4a_3x3_reduce = tf.nn.relu(conv3d(pool3, weights['W_inception_4a_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_4a_3x3_reduce'])
    inception_4a_3x3 = tf.nn.relu(conv3d(inception_4a_3x3_reduce, weights['W_inception_4a_3x3'],[1,1,1,1,1])+biases['b_inception_4a_3x3'])

    inception_4a_5x5_reduce = tf.nn.relu(conv3d(pool3, weights['W_inception_4a_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_4a_5x5_reduce'])
    inception_4a_5x5 = tf.nn.relu(conv3d(inception_4a_5x5_reduce, weights['W_inception_4a_5x5'],[1,1,1,1,1])+biases['b_inception_4a_5x5'])

    inception_4a_pool = maxpool3d(pool3,[1,3,3,3,1],[1,1,1,1,1])
    inception_4a_pool_proj = tf.nn.relu(conv3d(inception_4a_pool, weights['W_inception_4a_pool_proj'],[1,1,1,1,1])+biases['b_inception_4a_pool_proj'])

    inception_4a_output = depthconcat([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj])

    ### inception(4b)
    inception_4b_1x1 = tf.nn.relu(conv3d(inception_4a_output, weights['W_inception_4b_1x1'],[1,1,1,1,1])+biases['b_inception_4b_1x1'])

    inception_4b_3x3_reduce = tf.nn.relu(conv3d(inception_4a_output, weights['W_inception_4b_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_4b_3x3_reduce'])
    inception_4b_3x3 = tf.nn.relu(conv3d(inception_4b_3x3_reduce, weights['W_inception_4b_3x3'],[1,1,1,1,1])+biases['b_inception_4b_3x3'])

    inception_4b_5x5_reduce = tf.nn.relu(conv3d(inception_4a_output, weights['W_inception_4b_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_4b_5x5_reduce'])
    inception_4b_5x5 = tf.nn.relu(conv3d(inception_4b_5x5_reduce, weights['W_inception_4b_5x5'],[1,1,1,1,1])+biases['b_inception_4b_5x5'])

    inception_4b_pool = maxpool3d(inception_4a_output,[1,3,3,3,1],[1,1,1,1,1])
    inception_4b_pool_proj = tf.nn.relu(conv3d(inception_4b_pool, weights['W_inception_4b_pool_proj'],[1,1,1,1,1])+biases['b_inception_4b_pool_proj'])

    inception_4b_output = depthconcat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj])

    ### inception(4c)
    inception_4c_1x1 = tf.nn.relu(conv3d(inception_4b_output, weights['W_inception_4c_1x1'],[1,1,1,1,1])+biases['b_inception_4c_1x1'])

    inception_4c_3x3_reduce = tf.nn.relu(conv3d(inception_4b_output, weights['W_inception_4c_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_4c_3x3_reduce'])
    inception_4c_3x3 = tf.nn.relu(conv3d(inception_4c_3x3_reduce, weights['W_inception_4c_3x3'],[1,1,1,1,1])+biases['b_inception_4c_3x3'])

    inception_4c_5x5_reduce = tf.nn.relu(conv3d(inception_4b_output, weights['W_inception_4c_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_4c_5x5_reduce'])
    inception_4c_5x5 = tf.nn.relu(conv3d(inception_4c_5x5_reduce, weights['W_inception_4c_5x5'],[1,1,1,1,1])+biases['b_inception_4c_5x5'])

    inception_4c_pool = maxpool3d(inception_4b_output,[1,3,3,3,1],[1,1,1,1,1])
    inception_4c_pool_proj = tf.nn.relu(conv3d(inception_4c_pool, weights['W_inception_4c_pool_proj'],[1,1,1,1,1])+biases['b_inception_4c_pool_proj'])

    inception_4c_output = depthconcat([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj])

    ### inception(4d)
    inception_4d_1x1 = tf.nn.relu(conv3d(inception_4c_output, weights['W_inception_4d_1x1'],[1,1,1,1,1])+biases['b_inception_4d_1x1'])

    inception_4d_3x3_reduce = tf.nn.relu(conv3d(inception_4c_output, weights['W_inception_4d_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_4d_3x3_reduce'])
    inception_4d_3x3 = tf.nn.relu(conv3d(inception_4d_3x3_reduce, weights['W_inception_4d_3x3'],[1,1,1,1,1])+biases['b_inception_4d_3x3'])

    inception_4d_5x5_reduce = tf.nn.relu(conv3d(inception_4c_output, weights['W_inception_4d_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_4d_5x5_reduce'])
    inception_4d_5x5 = tf.nn.relu(conv3d(inception_4d_5x5_reduce, weights['W_inception_4d_5x5'],[1,1,1,1,1])+biases['b_inception_4d_5x5'])

    inception_4d_pool = maxpool3d(inception_4c_output,[1,3,3,3,1],[1,1,1,1,1])
    inception_4d_pool_proj = tf.nn.relu(conv3d(inception_4d_pool, weights['W_inception_4d_pool_proj'],[1,1,1,1,1])+biases['b_inception_4d_pool_proj'])

    inception_4d_output = depthconcat([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj])

    ### inception(4e)
    inception_4e_1x1 = tf.nn.relu(conv3d(inception_4d_output, weights['W_inception_4e_1x1'],[1,1,1,1,1])+biases['b_inception_4e_1x1'])

    inception_4e_3x3_reduce = tf.nn.relu(conv3d(inception_4d_output, weights['W_inception_4e_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_4e_3x3_reduce'])
    inception_4e_3x3 = tf.nn.relu(conv3d(inception_4e_3x3_reduce, weights['W_inception_4e_3x3'],[1,1,1,1,1])+biases['b_inception_4e_3x3'])

    inception_4e_5x5_reduce = tf.nn.relu(conv3d(inception_4d_output, weights['W_inception_4e_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_4e_5x5_reduce'])
    inception_4e_5x5 = tf.nn.relu(conv3d(inception_4e_5x5_reduce, weights['W_inception_4e_5x5'],[1,1,1,1,1])+biases['b_inception_4e_5x5'])

    inception_4e_pool = maxpool3d(inception_4d_output,[1,3,3,3,1],[1,1,1,1,1])
    inception_4e_pool_proj = tf.nn.relu(conv3d(inception_4e_pool, weights['W_inception_4e_pool_proj'],[1,1,1,1,1])+biases['b_inception_4e_pool_proj'])

    inception_4e_output = depthconcat([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj])
    pool4  = maxpool3d(inception_4e_output,[1,3,3,3,1],[1,2,2,2,1])

    ### inception(5a)
    inception_5a_1x1 = tf.nn.relu(conv3d(pool4, weights['W_inception_5a_1x1'],[1,1,1,1,1])+biases['b_inception_5a_1x1'])

    inception_5a_3x3_reduce = tf.nn.relu(conv3d(pool4, weights['W_inception_5a_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_5a_3x3_reduce'])
    inception_5a_3x3 = tf.nn.relu(conv3d(inception_5a_3x3_reduce, weights['W_inception_5a_3x3'],[1,1,1,1,1])+biases['b_inception_5a_3x3'])

    inception_5a_5x5_reduce = tf.nn.relu(conv3d(pool4, weights['W_inception_5a_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_5a_5x5_reduce'])
    inception_5a_5x5 = tf.nn.relu(conv3d(inception_5a_5x5_reduce, weights['W_inception_5a_5x5'],[1,1,1,1,1])+biases['b_inception_5a_5x5'])

    inception_5a_pool = maxpool3d(pool4,[1,3,3,3,1],[1,1,1,1,1])
    inception_5a_pool_proj = tf.nn.relu(conv3d(inception_5a_pool, weights['W_inception_5a_pool_proj'],[1,1,1,1,1])+biases['b_inception_5a_pool_proj'])

    inception_5a_output = depthconcat([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj])

    ### inception(5b)
    inception_5b_1x1 = tf.nn.relu(conv3d(inception_5a_output, weights['W_inception_5b_1x1'],[1,1,1,1,1])+biases['b_inception_5b_1x1'])

    inception_5b_3x3_reduce = tf.nn.relu(conv3d(inception_5a_output, weights['W_inception_5b_3x3_reduce'],[1,1,1,1,1])+biases['b_inception_5b_3x3_reduce'])
    inception_5b_3x3 = tf.nn.relu(conv3d(inception_5b_3x3_reduce, weights['W_inception_5b_3x3'],[1,1,1,1,1])+biases['b_inception_5b_3x3'])

    inception_5b_5x5_reduce = tf.nn.relu(conv3d(inception_5a_output, weights['W_inception_5b_5x5_reduce'],[1,1,1,1,1])+biases['b_inception_5b_5x5_reduce'])
    inception_5b_5x5 = tf.nn.relu(conv3d(inception_5b_5x5_reduce, weights['W_inception_5b_5x5'],[1,1,1,1,1])+biases['b_inception_5b_5x5'])

    inception_5b_pool = maxpool3d(inception_5a_output,[1,3,3,3,1],[1,1,1,1,1])
    inception_5b_pool_proj = tf.nn.relu(conv3d(inception_5b_pool, weights['W_inception_5b_pool_proj'],[1,1,1,1,1])+biases['b_inception_5b_pool_proj'])

    inception_5b_output = depthconcat([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj])

    ### average pool & dropout
    pool5 = avgpool3d(inception_5b_output,[1,7,7,7,1],[1,1,1,1,1])

    ###fc-relu-dropout
    fc = tf.nn.relu(tf.matmul(dropout1, weights['W_fc'])+biases['b_fc'])
    dropout1 = tf.nn.dropout(fc, keep_rate)
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

much_data = data
# If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-100]
validation_data = much_data[-100:]
import time

def train_neural_network(x):
    prediction = google_net(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

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
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
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

train_neural_network(x)