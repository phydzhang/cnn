# cnn
Several cnn networks for single particle 3D cryo-EM classification.

1.cnn_l2.py which has 3 conv2d layers, 3 max_pool layers and 3 full-connective layers uses l2 normalization of tensorflow module tf.nn.l2_loss

2.cnn_nor0.8 uses l2 normalization written by myself. 0.8 is the relatively better l2-normalization parameter "lamda"

3.vggnet.py is the vgg16 neural network written by myself.
