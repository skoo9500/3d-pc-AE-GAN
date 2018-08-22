import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import tf_util
from nn_distance import tf_nndistance
from transform_nets import input_transform_net, feature_transform_net
import numpy as np
from grouping.tf_grouping import query_ball_point, group_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    vector = tf.placeholder(tf.float32, shape=(batch_size, 1024))
    return pointclouds_pl, vector


def Encoder(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])
    net = tf_util.conv2d(net_transformed, 64, [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1], bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # Symmetric function: max pooling
    vector = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
    vector = tf.reshape(vector, [batch_size,-1])
    print(np.shape(vector))
    return vector

def Decoder(vector, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = vector.get_shape()[0].value
    print(np.shape(vector))
    net = tf.reshape(vector, [batch_size, 1, 1, -1])
    net = tf_util.conv2d_transpose(net, 512, kernel_size=[2, 2], stride=[2, 2], padding='VALID', scope='upconv2',
                                   bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[3, 3], stride=[1, 1], padding='VALID', scope='upconv3',
                                   bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[4, 4], stride=[2, 2], padding='VALID', scope='upconv4',
                                   bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 128, kernel_size=[5, 5], stride=[3, 3], padding='VALID', scope='upconv5',
                                   bn=True, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 3, kernel_size=[1, 1], stride=[1, 1], padding='VALID', scope='upconv6',
                                   activation_fn=None)
    net = tf.reshape(net, [batch_size, -1, 3])
    pred = net

    return pred

def Discriminator(point_cloud, is_training, bn_decay, reuse = False):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    with tf.variable_scope("dis", reuse=True):
        with tf.variable_scope('dtransform_net1', reuse=True) as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope='dconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope='dconv2', bn_decay=bn_decay)

        with tf.variable_scope('dtransform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])
        net = tf_util.conv2d(net_transformed, 64, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope='dconv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope='dconv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training,
                             scope='dconv5', bn_decay=bn_decay)
        # Symmetric function: max pooling
        vector = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='dmaxpool')
        vector = tf.reshape(vector, [batch_size, -1])
        net = tf.reshape(vector, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp3')
        net = tf_util.fully_connected(net, 1, activation_fn=None, scope='fc4')
        D_value= tf.reduce_mean(net)
    return D_value

def get_repulsion_loss(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss

def get_loss(pred, label):
    """ pred: BxNx3,
        label: BxNx3, """
    floss,_,bloss,_ = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(floss+bloss)

    return loss*100


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = Encoder(inputs, tf.constant(True))
        outputs = Decoder(outputs, tf.constant(True))

        print(outputs)
