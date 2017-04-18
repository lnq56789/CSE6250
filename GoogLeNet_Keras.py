
from scipy.misc import imread, imresize
import numpy as np
import os
import sys

from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Layer
from keras import backend as K
import theano.tensor as T

sys.stdout.flush()

K.set_image_dim_ordering('th')

class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:,:,1:,1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# input data
# data_paths = os.listdir("../Data/AllData12812848/")
data_list = ['alldata3-128-128-80-1000.npy', 'alldata3-128-128-80-1200.npy',
             'alldata3-128-128-80-1400.npy', 'alldata3-128-128-80-1594.npy',
             'alldata3-128-128-80-200.npy', 'alldata3-128-128-80-400.npy',
             'alldata3-128-128-80-600.npy', 'alldata3-128-128-80-800.npy']
data_all = []
for path in data_list:
    patial_data = np.load(path)
    data_all.append(patial_data)

data = np.concatenate(data_all)

x = np.stack(data[:,0], axis=0)
y = np.stack(data[:,1], axis=0)


input = Input(shape=(80, 128, 128))

conv1_7x7_s2 = Conv2D(64,7,strides=(2,2),padding='same',activation='relu',name='conv1/7x7_s2',W_regularizer=l2(0.0002))(input)

conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

pool1_helper = PoolHelper()(conv1_zero_pad)

pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool1/3x3_s2')(pool1_helper)

pool1_norm1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, name='pool1/norm1')(pool1_3x3_s2)

conv2_3x3_reduce = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='conv2/3x3_reduce',W_regularizer=l2(0.0002))(pool1_norm1)

conv2_3x3 = Conv2D(192,3,strides=(1,1),padding='same',activation='relu',name='conv2/3x3',W_regularizer=l2(0.0002))(conv2_3x3_reduce)

conv2_norm2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, name='conv2/norm2')(conv2_3x3)

conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)

pool2_helper = PoolHelper()(conv2_zero_pad)

pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool2/3x3_s2')(pool2_helper)


inception_3a_1x1 = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_3x3_reduce = Conv2D(96,1,strides=(1,1),padding='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_3x3 = Conv2D(128,3,strides=(1,1),padding='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)

inception_3a_5x5_reduce = Conv2D(16,1,strides=(1,1),padding='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002))(pool2_3x3_s2)

inception_3a_5x5 = Conv2D(32,5,strides=(1,1),padding='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)

inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3a/pool')(pool2_3x3_s2)

inception_3a_pool_proj = Conv2D(32,1,strides=(1,1),padding='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002))(inception_3a_pool)

inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],mode='concat',concat_axis=1,name='inception_3a/output')


inception_3b_1x1 = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_3b/1x1',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_3x3_reduce = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_3b/3x3_reduce',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_3x3 = Conv2D(192,3,strides=(1,1),padding='same',activation='relu',name='inception_3b/3x3',W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

inception_3b_5x5_reduce = Conv2D(32,1,strides=(1,1),padding='same',activation='relu',name='inception_3b/5x5_reduce',W_regularizer=l2(0.0002))(inception_3a_output)

inception_3b_5x5 = Conv2D(96,5,strides=(1,1),padding='same',activation='relu',name='inception_3b/5x5',W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)

inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_3b/pool')(inception_3a_output)

inception_3b_pool_proj = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_3b/pool_proj',W_regularizer=l2(0.0002))(inception_3b_pool)

inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],mode='concat',concat_axis=1,name='inception_3b/output')


inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool3/3x3_s2')(pool3_helper)


inception_4a_1x1 = Conv2D(192,1,strides=(1,1),padding='same',activation='relu',name='inception_4a/1x1',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_3x3_reduce = Conv2D(96,1,strides=(1,1),padding='same',activation='relu',name='inception_4a/3x3_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_3x3 = Conv2D(208,3,strides=(1,1),padding='same',activation='relu',name='inception_4a/3x3',W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)

inception_4a_5x5_reduce = Conv2D(16,1,strides=(1,1),padding='same',activation='relu',name='inception_4a/5x5_reduce',W_regularizer=l2(0.0002))(pool3_3x3_s2)

inception_4a_5x5 = Conv2D(48,5,strides=(1,1),padding='same',activation='relu',name='inception_4a/5x5',W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4a/pool')(pool3_3x3_s2)

inception_4a_pool_proj = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_4a/pool_proj',W_regularizer=l2(0.0002))(inception_4a_pool)

inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],mode='concat',concat_axis=1,name='inception_4a/output')


inception_4b_1x1 = Conv2D(160,1,strides=(1,1),padding='same',activation='relu',name='inception_4b/1x1',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_3x3_reduce = Conv2D(112,1,strides=(1,1),padding='same',activation='relu',name='inception_4b/3x3_reduce',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_3x3 = Conv2D(224,3,strides=(1,1),padding='same',activation='relu',name='inception_4b/3x3',W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)

inception_4b_5x5_reduce = Conv2D(24,1,strides=(1,1),padding='same',activation='relu',name='inception_4b/5x5_reduce',W_regularizer=l2(0.0002))(inception_4a_output)

inception_4b_5x5 = Conv2D(64,5,strides=(1,1),padding='same',activation='relu',name='inception_4b/5x5',W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)

inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4b/pool')(inception_4a_output)

inception_4b_pool_proj = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_4b/pool_proj',W_regularizer=l2(0.0002))(inception_4b_pool)

inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],mode='concat',concat_axis=1,name='inception_4b_output')


inception_4c_1x1 = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_4c/1x1',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_3x3_reduce = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_4c/3x3_reduce',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_3x3 = Conv2D(256,3,strides=(1,1),padding='same',activation='relu',name='inception_4c/3x3',W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)

inception_4c_5x5_reduce = Conv2D(24,1,strides=(1,1),padding='same',activation='relu',name='inception_4c/5x5_reduce',W_regularizer=l2(0.0002))(inception_4b_output)

inception_4c_5x5 = Conv2D(64,5,strides=(1,1),padding='same',activation='relu',name='inception_4c/5x5',W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)

inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4c/pool')(inception_4b_output)

inception_4c_pool_proj = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_4c/pool_proj',W_regularizer=l2(0.0002))(inception_4c_pool)

inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],mode='concat',concat_axis=1,name='inception_4c/output')


inception_4d_1x1 = Conv2D(112,1,strides=(1,1),padding='same',activation='relu',name='inception_4d/1x1',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_3x3_reduce = Conv2D(144,1,strides=(1,1),padding='same',activation='relu',name='inception_4d/3x3_reduce',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_3x3 = Conv2D(288,3,strides=(1,1),padding='same',activation='relu',name='inception_4d/3x3',W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)

inception_4d_5x5_reduce = Conv2D(32,1,strides=(1,1),padding='same',activation='relu',name='inception_4d/5x5_reduce',W_regularizer=l2(0.0002))(inception_4c_output)

inception_4d_5x5 = Conv2D(64,5,strides=(1,1),padding='same',activation='relu',name='inception_4d/5x5',W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)

inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4d/pool')(inception_4c_output)

inception_4d_pool_proj = Conv2D(64,1,strides=(1,1),padding='same',activation='relu',name='inception_4d/pool_proj',W_regularizer=l2(0.0002))(inception_4d_pool)

inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],mode='concat',concat_axis=1,name='inception_4d/output')


inception_4e_1x1 = Conv2D(256,1,strides=(1,1),padding='same',activation='relu',name='inception_4e/1x1',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_3x3_reduce = Conv2D(160,1,strides=(1,1),padding='same',activation='relu',name='inception_4e/3x3_reduce',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_3x3 = Conv2D(320,3,strides=(1,1),padding='same',activation='relu',name='inception_4e/3x3',W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)

inception_4e_5x5_reduce = Conv2D(32,1,strides=(1,1),padding='same',activation='relu',name='inception_4e/5x5_reduce',W_regularizer=l2(0.0002))(inception_4d_output)

inception_4e_5x5 = Conv2D(128,5,strides=(1,1),padding='same',activation='relu',name='inception_4e/5x5',W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)

inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_4e/pool')(inception_4d_output)

inception_4e_pool_proj = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_4e/pool_proj',W_regularizer=l2(0.0002))(inception_4e_pool)

inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],mode='concat',concat_axis=1,name='inception_4e/output')


inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',name='pool4/3x3_s2')(pool4_helper)


inception_5a_1x1 = Conv2D(256,1,strides=(1,1),padding='same',activation='relu',name='inception_5a/1x1',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_3x3_reduce = Conv2D(160,1,strides=(1,1),padding='same',activation='relu',name='inception_5a/3x3_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_3x3 = Conv2D(320,3,strides=(1,1),padding='same',activation='relu',name='inception_5a/3x3',W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)

inception_5a_5x5_reduce = Conv2D(32,1,strides=(1,1),padding='same',activation='relu',name='inception_5a/5x5_reduce',W_regularizer=l2(0.0002))(pool4_3x3_s2)

inception_5a_5x5 = Conv2D(128,5,strides=(1,1),padding='same',activation='relu',name='inception_5a/5x5',W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5a/pool')(pool4_3x3_s2)

inception_5a_pool_proj = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_5a/pool_proj',W_regularizer=l2(0.0002))(inception_5a_pool)

inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],mode='concat',concat_axis=1,name='inception_5a/output')


inception_5b_1x1 = Conv2D(384,1,strides=(1,1),padding='same',activation='relu',name='inception_5b/1x1',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_3x3_reduce = Conv2D(192,1,strides=(1,1),padding='same',activation='relu',name='inception_5b/3x3_reduce',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_3x3 = Conv2D(384,3,strides=(1,1),padding='same',activation='relu',name='inception_5b/3x3',W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)

inception_5b_5x5_reduce = Conv2D(48,1,strides=(1,1),padding='same',activation='relu',name='inception_5b/5x5_reduce',W_regularizer=l2(0.0002))(inception_5a_output)

inception_5b_5x5 = Conv2D(128,5,strides=(1,1),padding='same',activation='relu',name='inception_5b/5x5',W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)

inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same',name='inception_5b/pool')(inception_5a_output)

inception_5b_pool_proj = Conv2D(128,1,strides=(1,1),padding='same',activation='relu',name='inception_5b/pool_proj',W_regularizer=l2(0.0002))(inception_5b_pool)

inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],mode='concat',concat_axis=1,name='inception_5b/output')


pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),padding='same',strides=(1,1),name='pool5/7x7_s2')(inception_5b_output)

loss3_flat = Flatten()(pool5_7x7_s1)

pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

loss3_classifier = Dense(2,name='loss3/classifier',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)

loss3_classifier_act = Activation('softmax',name='prob')(loss3_classifier)


googlenet = Model(input=input, output=[loss3_classifier_act])
googlenet.summary()

# adagrad = Adagrad(lr=0.01, epsilon=1e-06)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
googlenet.compile(optimizer=sgd, loss='binary_crossentropy',metrics=['accuracy'])
trained = googlenet.fit(x,y, batch_size=128,epochs=1000, class_weight={1:0.7409, 0:0.2591})




