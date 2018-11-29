#This program is to use vggnet to 

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
    y_pre = sess.run(predictions, feed_dict={x_images: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={ys: v_ys, keep_prob: 1})
    return result


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
         kernel        = tf.get_variable(scope+"w",
                           shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
         conv          = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
         bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
         biases        = tf.Variable(bias_init_val, trainable=True, name='b')
         z             = tf.nn.bias_add(conv, biases)
         activation    = tf.nn.relu(z, name=scope)
         p            += [kernel, biases]
         return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
         kernel     = tf.get_variable(scope+"w",
                        shape=[n_in, n_out], dtype=tf.float32,
                        initializer=tf.contrib.layers.xavier_initializer())
         biases     = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
         activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
         p         += [kernel, biases]
         return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, 
                          ksize=[1, kh, kw, 1], 
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


#define placeholder for inputs to network
x_images = tf.placeholder(tf.float32, [None,300,300,1])/255.  #200x200
xs = tf.placeholder(tf.float32, [None, 90000])/255.
ys = tf.placeholder(tf.float32, [None,2])
keep_prob = tf.placeholder(tf.float32)


#Define the structure of the net

p = []

pool0 = mpool_op(x_images, name="pool0", kh=2, kw=2, dw=2, dh=2)

conv1_1 = conv_op(pool0, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)

conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)

pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)


conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)

conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)

pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dw=2, dh=2)


conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)

conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)

conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)

pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dw=2, dh=2)


conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dw=2, dh=2)


conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)

pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)


shp = pool5.get_shape()
flattened_shape = shp[1].value * shp[2].value * shp[3].value
resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")


fc6 = fc_op(resh1, name="fc6", n_out=1024, p=p)
fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

fc7 = fc_op(fc6_drop, name="fc7", n_out=512, p=p)
fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

fc8 = fc_op(fc7_drop, name="fc8", n_out=2, p=p)
predictions = tf.nn.softmax(fc8)
#predictions = tf.argmax(softmax, 1)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(predictions + 1e-10),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)



#Setting the running environment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

saver = tf.train.Saver(max_to_keep=5)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#saver.restore(sess, "./saver_vggnet/vggnet.ckpt")

doc=open('./out/out_vggnet.txt','w')

batch_size = 100
acc_test = 0.0
acc_train = 0.0


#Run the train_step
for i in range(300):

#    rand = np.arange(16000)
#    permutation = np.random.permutation(rand.shape[0])
#    train_images_rand = train_images[permutation,:,:,:]
#    train_labels_rand = train_labels[permutation,:]

    for j in range(160):
       train_x = train_images[batch_size*i:batch_size*(i+1)]
       train_y = train_labels[batch_size*i:batch_size*(i+1)]
       sess.run(train_step, feed_dict={x_images: train_x, ys: train_y, keep_prob: 0.5})

    for k in range(26):
       test_x = test_images[97*k:97*(k+1)]
       test_y = test_labels[97*k:97*(k+1)]
       acc_test += compute_accuracy(test_x, test_y)
    acc_test = acc_test/26.0

    for k in range(160):
       train_x0 = train_images[100*k:100*(k+1)]
       train_y0 = train_labels[100*k:100*(k+1)]
       acc_train = acc_train + compute_accuracy(train_x0, train_y0)
    acc_train = acc_train/160.0


    print("The",i,"th epoch: ",round(acc_test,6),"   ",round(acc_train,6),file=doc)
    print("The",i,"th epoch: ",round(acc_test,6),"   ",round(acc_train,6))
    save_path = saver.save(sess, "./saver_vggnet/vggnet.ckpt", global_step=i)


doc.close()









