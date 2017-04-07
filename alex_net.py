# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 21:20:23 2017

@author: Zimu, Hanying
"""

# import dicom # for reading dicom files
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
#Below is code to load a scan, which consists of multiple slices, which we simply save in a Python list.
#Every folder in the dataset is one scan (so one patient). One metadata field is missing, the pixel size in the Z direction,
#which is the slice thickness. Fortunately we can infer this, and we add this to the metadata

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
#
    return slices

#Some scanners have cylindrical scanning bounds, but the output image is square.
#The pixels that fall outside of these bounds get the fixed value -2000.
#The first step is setting these values to 0, which currently corresponds to air.
#Next, let's go back to HU units, by multiplying with the rescale slope and adding the intercept
#(which are conveniently stored in the metadata of the scans!).
def get_pixels_hu(slices):
#     credit to Guido Zuidhof https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

#    A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters.
#    For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets)!
#    A common method of dealing with this is resampling the full dataset to a certain isotropic resolution.
#    If we choose to resample everything to 1mm1mm1mm pixels we can use 3D convnets without worrying about
#    learning zoom/slice thickness invariance.

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def plot_3d(image, threshold=-300):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

#Threshold the image (-320 HU is a good threshold, but it doesn't matter much for this approach)
#Do connected components, determine label of air around person, fill this with 1s in the binary image
#Optionally: For every axial slice in the scan, determine the largest solid connected component (the body+air around the person), and set others to 0. This fills the structures in the lungs in the mask.
#Keep only the largest air pocket (the human body has other pockets of air here and there).
def largest_label_volume(im, bg=-1):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0,0,0]

    #Fill the air around the person
    binary_image[background_label == labels] = 2


    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1


    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0

    return binary_image




def normalize(image,MIN_BOUND = -1000.0,MAX_BOUND = 400.0):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def zero_center(image,PIXEL_MEAN = 0.25):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    image = image - PIXEL_MEAN
    return image
def chunks(l, n,HM_SLICES = 20):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    count=0
    for i in range(0, len(l), n):
        if(count < HM_SLICES):
            yield l[i:i + n]
            count=count+1

def mean(l):
     #    credit to Guido Zuidhof  https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    return sum(l) / len(l)
def process_data(patient,labels_df,data_dir,img_px_size=50,  hm_slices=20, visualize=False, superPixels = 100):

    #lung segmentation
    label = labels_df.get_value(patient, 'cancer')
    path = data_dir + patient
    slices = load_scan(path)
    if (len(slices)==0):
        print("No Data")
        return ([0, 0])
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    slices_pixels = get_pixels_hu(slices)

    segmented_lungs_fill = segment_lung_mask(np.asarray(slices_pixels), True)
#    superPixelImage = slic(segmented_lungs_fill,n_segments=superPixels,compactness = 0.04, multichannel = False)
#    pix_resampled = resample(slices_pixels, slices, [1,1,1])
    newSlicesPixels = []
    for each_slice in range(0,segmented_lungs_fill.shape[0]):
#        superPixelImage = slic(segmented_lungs_fill[each_slice],n_segments=superPixels,compactness = 0.04, multichannel = False)
#        superPixelImage = scipy.ndimage.filters.gaussian_filter(superPixelImage,.5)
#        superPixMax = (np.amax(superPixelImage[each_slice]))
#        im = img.fromarray(np.uint8(superPixelImage/superPixMax)*255).resize((img_px_size,img_px_size))
#        im = np.fromstring(im.tobytes(),dtype=np.uint8)/255*superPixMax.reshape(img_px_size,img_px_size)
        im = img.fromarray(np.uint8(segmented_lungs_fill[each_slice])*255).resize((img_px_size,img_px_size))
        im = (np.fromstring(im.tobytes(),dtype=np.uint8)/255).reshape(img_px_size,img_px_size)
        newSlicesPixels.append(im)

    newSlicesPixels = np.asarray(newSlicesPixels)

    new_slices = []
    chunk_sizes = math.floor(newSlicesPixels.shape[0] / hm_slices)
    for slice_chunk in chunks(newSlicesPixels, int(chunk_sizes),hm_slices):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()

    if label == 1: label=np.array([0,1])
    elif label == 0: label=np.array([1,0])

    return np.array(new_slices),label

#tf.nn.conv3d(input, filter, strides, padding, name=None)
def batchnorm_layer(Ylogits, is_test, Offset, Scale, iteration, convolutional=False):
#credit to Martin Gorner https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.2_batchnorm_convolutional.py
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999,iteration)
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2, 3])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.averge(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda:variance)
    Ybn = tf.nn.batch_normalization(Ylogits,m,v,Offset,Scale,variance_epsilon=1e-5)
    return Ybn, update_moving_averages
#def conv3d(x, W, padding='Same'):
#    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')
#tf.nn.conv3d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
def conv3dT(x, W, outputShape, padding='Same'):
    return tf.nn.conv3d_transpose(x, W, outputShape,strides=[1,2,2,2,1], padding='SAME')

def conv3d(x, W, s):
    return tf.nn.conv3d(x, W, strides=s, padding='SAME')

def maxpool3d(x):
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
    weights = {'W_conv1':tf.Variable(tf.random_normal([11,11,11,1,96],stddev=1/math.sqrt(121*11))),
               'W_conv2':tf.Variable(tf.random_normal([5,5,5,96,256],stddev=1/math.sqrt(125*96))),
               'W_conv3':tf.Variable(tf.random_normal([3,3,3,256,384],stddev=1/math.sqrt(27*256))),
               'W_conv4':tf.Variable(tf.random_normal([3,3,3,384,384],stddev=1/math.sqrt(27*384))),
               'W_conv5':tf.Variable(tf.random_normal([3,3,3,384,256],stddev=1/math.sqrt(27*384))),
               'W_fc1':tf.Variable(tf.random_normal([602112,4096],stddev=1/math.sqrt(27))),
               'W_fc2':tf.Variable(tf.random_normal([4096,4096],stddev=1/math.sqrt(4096))),
               'out':tf.Variable(tf.random_normal([4096, n_classes],stddev=1/math.sqrt(4096)))}

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
    prediction = AlexNet(x,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
#    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#   apply gradient clipping to eliminate gradient explosion problem
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    gradients =   optimizer.compute_gradients(cost)

    capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gvs)
    hm_epochs = 10


    with tf.Session() as sess:
        print("initializing sess")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        successful_runs = 0
        total_runs = 0
        print("beginning runs")
        for epoch in range(hm_epochs):
            error = 0
            epoch_loss = 0
            thisRun = 0
            for data in train_data:

                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([train_op, cost], feed_dict={x: X, y: Y})

                    epoch_loss += c
#                    test_error = error_rate(eval_in_batches(testX, sess,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate), testY)
#                    print(test_error)
                    successful_runs += 1
                    thisRun += 1
                    if thisRun % 100 == 0:
                        print(str(thisRun) + " out of " + str(len(train_data)))
                        print("run:"+str(thisRun)+" with epoch_loss"+str(epoch_loss) + "with loss c:" + str(c))
                        saver.save(sess, './unetmodel')
                    elif thisRun %5  == 0:
                        print("run:"+str(thisRun)+" with epoch_loss"+str(epoch_loss) + "with loss c:" + str(c))
                        saver.save(sess, './unetmodel')
                except Exception as e:
                    # I am passing for the sake of notebook space, but we are getting 1 shaping issue from one
                    # input tensor. Not sure why, will have to look into it. Guessing it's
                    # one of the depths that doesn't come to 20.
                    error += 1
                    if error % 100 == 0:
                        print("error:"+str(error) + " out of " + str(len(train_data)))
                    pass
                    #print(str(e))
            saver.save(sess, './unetmodel')
            print("error:"+str(error))
            print("successful_runs:"+str(successful_runs))
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        saver.save(sess, 'unet-model',global_step = successful_runs)
#            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#
#            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('Done. Finishing accuracy:')
#        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))

        print('fitment percent:',successful_runs/total_runs)

def main():



#    At this point, we've got the list of patients by their IDs, and their associated labels stored in a dataframe.
#    Now, we can begin to iterate through the patients and gather their respective data.
#    We're almost certainly going to need to do some preprocessing of this data, but we'll see.

    x = tf.placeholder('float')
    y = tf.placeholder('float')




    #data_dir = 'C:/Users/Zimu/Desktop/Project/TrainingData/'
    if processData:
        data_dir = 'D:/S2/stage1/'

        patients = os.listdir(data_dir)
        patients.sort()
        patients = os.listdir(data_dir)
        labels = pd.read_csv('C:/Users/Zimu/Desktop/Project/stage1_labels.csv', index_col=0)
        numSuperPixels=100
        much_data = []
        for num,patient in enumerate(patients):
            print(str(num) + ":" + patient)
            if num % 100 == 0:
                print(str(num) + ":" + patient)
            try:
                img_data,label = process_data(patient,labels,data_dir, img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT, superPixels = numSuperPixels)
    #            np.save('imgData-{}-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT,patient), img_data)
                #print(img_data.shape,label)
                if (1-np.isscalar(img_data)):
                    much_data.append([img_data,label])
            except KeyError as e:
                print('This is unlabeled data!')
        np.save('alldata3-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)

    much_data1 = np.load('alldata3-64-64-24.npy')

    train_data1 = much_data1[:-300]
    validation_data1 = much_data1[-300:]




#    train_neural_network(x,y,train_data1,validation_data1,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)
    train_alex_net(x,y,train_data1,validation_data1,n_classes,IMG_SIZE_PX,SLICE_COUNT,keep_rate)


if __name__ == "__main__":

	main()
