#This program is to use cnn to classify the cryo-EM image 
#with no noise to env5 & env6


from __future__ import print_function
import tensorflow as tf
import numpy as np
import os

#input data from path
train_images = np.load("../en5_en6_19s_bin2/train_x_en5_en6_19s_300_16000_bin2.npy")
train_labels = np.load("../en5_en6_19s_bin2/train_y_en5_en6_19s_300_16000_bin2.npy")
test_images = np.load("../en5_en6_19s_bin2/test_x_en5_en6_19s_300_2522_bin2.npy")
test_labels = np.load("../en5_en6_19s_bin2/test_y_en5_en6_19s_300_2522_bin2.npy")

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x_images: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={ys: v_ys, keep_prob: 1})
    return result



def weight_variable(shape, stddev, wl):
    initial = tf.truncated_normal(shape,stddev=stddev)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(initial),wl,name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return tf.Variable(initial, dtype=tf.float32)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)


def conv2d(x,W):
    # stride [1, x_movement, y_movement, 1]
    # must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    # stride [1,x_movement,y_movement,1], ksize [1, length_pool, hith_pool, 1]
    return tf.nn.max_pool(x, ksize=[1,5,5,1], strides=[1,3,3,1], padding='SAME')


#define placeholder for inputs to network
x_images = tf.placeholder(tf.float32, [None,300,300,1])/255.  #200x200
xs = tf.placeholder(tf.float32, [None, 90000])/255.
ys = tf.placeholder(tf.float32, [None,2])
keep_prob = tf.placeholder(tf.float32)
#x_images = tf.reshape(xs,[-1,200,200,1])

## pool0 layer ##
#h_pool0 = tf.nn.max_pool(x_images, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

## conv1 layer ##
W_conv1 = weight_variable(shape=[5,5,1,16], stddev=1e-2, wl=0.0) #patch 4x4, in size 1, out size 16
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1) #output size 200x200x16
h_pool1 = max_pool_2x2(h_conv1)    #output size 200x200x16



## conv2 layer ##
W_conv2 = weight_variable(shape=[5,5,16,32], stddev=1e-2, wl=0.0) #patch 4x4, in size 16, out size 32
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #output size 100x100x32
h_pool2 = max_pool_2x2(h_conv2)  #output size 67x67x32


## conv3 layer ##
W_conv3 = weight_variable(shape=[5,5,32,64], stddev=1e-2, wl=0.0) #patch 4x4, in size 32, out size 64
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) #output size 50x50x64
h_pool3 = max_pool_2x2(h_conv3)  #output size 12x12x64


## conv4 layer ##
#W_conv4 = weight_variable([5,5,64,128]) #patch 4x4, in size 64, out size 128
#b_conv4 = bias_variable([128])
#h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4) #output size 25x25x128
#h_pool4 = max_pool_2x2(h_conv4)  #output size 8x8x128


## conv5 layer ##
#W_conv5 = weight_variable([5,5,128,256]) #patch 4x4, in size 128, out size 256
#b_conv5 = bias_variable([256])
#h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5) #output size 13x13x256
#h_pool5 = max_pool_2x2(h_conv5)  #output size 7x7x256


## fc1 layer ##
W_fc1 = weight_variable(shape=[12*12*64, 512], stddev=0.01, wl=0.004)
b_fc1 = bias_variable([512])
# [n_samples, 50, 50, 64] ->> [n_samples, 50*50*64]
h_pool3_flat = tf.reshape(h_pool3, [-1,12*12*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


## fc2 layer ##
W_fc2 = weight_variable(shape=[512,256], stddev=0.01, wl=0.004)
b_fc2 = bias_variable([256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


## fc3 layer ##
W_fc3 = weight_variable(shape=[256,2], stddev=1/256.0, wl=0.0)
b_fc3 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction + 1e-8),reduction_indices=[1]))
tf.add_to_collection('losses', cross_entropy)
loss = tf.add_n(tf.get_collection('losses'), name='total_loss') + 1e-8

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

saver = tf.train.Saver(max_to_keep=5)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#saver.restore(sess, "./saver_cnnl2/cnnl2.ckpt")

doc=open('./out/out_cnnl2.txt','w')

batch_size = 100
acc_test = 0.0
acc_train = 0.0
ll = 0.0

for i in range(1000):
 
#    rand = np.arange(16000)
#    permutation = np.random.permutation(rand.shape[0])
#    train_images_rand = train_images[permutation,:,:,:]
#    train_labels_rand = train_labels[permutation,:]

    for j in range(160):
       train_x = train_images[batch_size*j:batch_size*(j+1)]
       train_y = train_labels[batch_size*j:batch_size*(j+1)]
       sess.run(train_step, feed_dict={x_images: train_x, ys: train_y, keep_prob: 0.5})
#       if j % 32 == 0:
#         for k in range(160):
#           train_x0 = train_images[100*k:100*(k+1)]
#           train_y0 = train_labels[100*k:100*(k+1)]
#           acc_train = acc_train + compute_accuracy(train_x0, train_y0)
#         acc_train = acc_train/160.0
#         print("acc_train=  ", acc_train)

    for k in range(26):
       test_x = test_images[97*k:97*(k+1)]
       test_y = test_labels[97*k:97*(k+1)]
       acc_test += compute_accuracy(test_x, test_y)
    acc_test = acc_test/26.0

    for k in range(160):
       train_x0 = train_images[100*k:100*(k+1)]
       train_y0 = train_labels[100*k:100*(k+1)]
       acc_train = acc_train + compute_accuracy(train_x0, train_y0)
       ll = ll + sess.run(cross_entropy, feed_dict={x_images: train_x0, ys: train_y0, keep_prob: 1})
    acc_train = acc_train/160.0
    ll = ll/160.0


    print("The",i,"th epoch: ",round(acc_test,6),"   ",round(acc_train,6),"loss = ",round(ll,6),file=doc)
    print("The",i,"th epoch: ",round(acc_test,6),"   ",round(acc_train,6),"loss = ",round(ll,6))
    save_path = saver.save(sess, "./saver_cnnl2/cnnl2.ckpt", global_step=i)



doc.close()













