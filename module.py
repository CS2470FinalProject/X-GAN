from __future__ import division
import tensorflow as tf

from ops import *
from utils import *


def discriminator(image, options, domain_name):
    
    with tf.variable_scope(domain_name, reuse=tf.AUTO_REUSE):
        # image is (64 x 64 x input_c_dim)
        h1 = lrelu(conv2d(image, options.dcf_dim, name = 'h1_conv'))
        # h1 is (32 x 32 x args.ndcf)
        h2 = lrelu(instance_norm(conv2d(h1, options.dcf_dim * 2, name = 'h2_conv_' + domain_name), 'h_bn2_' + domain_name))
        # h2 is (16 x 16 x args.ndcf * 2)
        h3 = lrelu(instance_norm(conv2d(h2, options.dcf_dim * 2, name='h3_conv_' + domain_name), 'h_bn3_' + domain_name))
        # h3 is (8 x 8 x args.ndcf * 2)
        h4 = lrelu(instance_norm(conv2d(h3, options.dcf_dim * 2, name='h4_conv_' + domain_name), 'h_bn4_' + domain_name))
        # h4 is (4 x 4 x args.ndcf * 2)
        h5 = h4.reshape(h4, [1, 1, -1])
        h5 = linear(h5, 1, scope = domain_name)
        # h5 is (1 x 1 x 1)

    return h5

def encoder(image, options, domain_name):
    
    with tf.variable_scope("encoder_" + domain_name, reuse=tf.AUTO_REUSE):
        # image is (64 x 64 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.ef_dim, name = 'e1_conv_' + domain_name), 'e_bn1_' + domain_name)
        # e1 is (32 x 32 x args.nef)
        e2 = instance_norm(conv2d(lrelu(e1), options.ef_dim * 2, name = 'e2_conv' + domain_name), 'e_bn2_' + domain_name)
        # e2 is (16 x 16 x args.nef * 2)

    with tf.variable_scope("encoder_sharing", reuse=tf.AUTO_REUSE):
        e3 = instance_norm(conv2d(lrelu(e2), options.ef_dim * 4, name = 'e3_conv'), 'e_bn3')
        # e3 is (8 x 8 x args.nef * 4)
        e4 = instance_norm(conv2d(lrelu(e3), options.ef_dim * 8, name = 'e4_conv'), 'e_bn4')
        # e4 is (4 x 4 x args.nef * 8)
        e4 = tf.reshape(e4, [1, 1, -1])
        # e4 is (1 x 1 x 4 * 4 * args.nef * 8)
        e5 = instance_norm(linear(e4, 1024), 'fc1')
        # e5 is (1 x 1 x 1024)
        e6 = instance_norm(linear(e5, 1024), 'fc2')
        # e6 is (1 x 1 x 1024)

    return e6

def decoder(image, options, domain_name):
    
    with tf.variable_scope("decoder_sharing", reuse=tf.AUTO_REUSE):
        d1 = tf.reshape(image, [2, 2, 256])
        d1 = instance_norm(deconv2d(d1, options.df_dim * 8, name = 'd1_dconv'), 'd_bn1')
        # d1 is (4 x 4 x args.ndf * 8)
        d2 = instance_norm(deconv2d(d1, options.df_dim * 4, name = 'd2_dconv'), 'd_bn2')
        # d2 is (8 x 8 x args.ndf * 4)

    
    with tf.variable_scope("decoder_" + domain_name, reuse=tf.AUTO_REUSE):
        d3 = instance_norm(deconv2d(d2, options.df_dim * 2, name='d3_dconv_' + domain_name), 'd_bn3_' + domain_name)
        # d3 is (16 x 16 x args.ndf * 2)
        d4 = instance_norm(deconv2d(d3, options.df_dim, name='d4_dconv_' + domain_name), 'd_bn4_' + domain_name)
        # d4 is (32 x 32 x args.ndf)
        d5 = instance_norm(deconv2d(d4, options.output_c_dim, name='d5_dconv_' + domain_name), 'd_bn5' + domain_name)
        # d5 is (64 x 64 x args.output_nc)

    return d5
        
def euc_criterion(in_, target):
    return tf.losses.mean_squared_error(target, in_)

    
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))