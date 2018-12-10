from __future__ import division
import tensorflow as tf

from ops import *
from utils import *

from flip_gradient import flip_gradient


def discriminator(image, options, name):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # image is (batch_size x 64 x 64 x input_c_dim)
        h1 = lrelu(conv2d(image, options.dcf_dim, name='h1_conv'))
        # h1 is (batch_size x 32 x 32 x args.ndcf)
        h2 = lrelu(instance_norm(conv2d(h1, options.dcf_dim * 2, name='h2_conv'), 'h_bn2'))
        # h2 is (batch_size x 16 x 16 x args.ndcf * 2)
        h3 = lrelu(instance_norm(conv2d(h2, options.dcf_dim * 2, name='h3_conv'), 'h_bn3'))
        # h3 is (batch_size x 8 x 8 x args.ndcf * 2)
        h4 = lrelu(instance_norm(conv2d(h3, options.dcf_dim * 2, name='h4_conv'), 'h_bn4'))
        # h4 is (batch_size x 4 x 4 x args.ndcf * 2)
        h5 = conv2d(h4, 1, ks=4, s=1, padding='VALID', name='d_h3_pred')
        # h5 is (batch_size x 1 x 1 x 1)

    return h5


def encoder(image, options, domain_name):
    
    with tf.variable_scope("encoder_" + domain_name, reuse=tf.AUTO_REUSE):
        # image is (batch_size x 64 x 64 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.ef_dim, name='e1_conv'), 'e_bn1')
        # e1 is (batch_size x 32 x 32 x args.nef)
        e2 = instance_norm(conv2d(lrelu(e1), options.ef_dim * 2, name='e2_conv'), 'e_bn2')
        # e2 is (batch_size x 16 x 16 x args.nef * 2)

    with tf.variable_scope("encoder_sharing", reuse=tf.AUTO_REUSE):
        e3 = instance_norm(conv2d(lrelu(e2), options.ef_dim * 4, name='e3_conv'), 'e_bn3')
        # e3 is (batch_size x 8 x 8 x args.nef * 4)
        e4 = instance_norm(conv2d(lrelu(e3), options.ef_dim * 8, name='e4_conv'), 'e_bn4')
        # e4 is (batch_size x 4 x 4 x args.nef * 8)
        e5 = conv2d(lrelu(e4), 1024, ks=4, s=1, padding='VALID', name='e5_conv')
        # e5 is (batch_size x 1 x 1 x 1024)
        e6 = conv2d(lrelu(e5), 1024, ks=1, s=1, padding='VALID', name='e6_conv')
        # e6 is (batch_size x 1 x 1 x 1024)

    return e6


def decoder(input, options, domain_name):
    
    with tf.variable_scope("decoder_sharing", reuse=tf.AUTO_REUSE):
        #d1 = tf.reshape(input, [tf.shape(input)[0], 2, 2, 256])
        d1 = instance_norm(deconv2d(tf.nn.relu(input), options.df_dim * 8, s=4, name='d1_dconv'), 'd_bn1')
        # d1 is (bathc_size x 4 x 4 x args.ndf * 8)
        d2 = instance_norm(deconv2d(tf.nn.relu(d1), options.df_dim * 4, name='d2_dconv'), 'd_bn2')
        # d2 is (batch_size x 8 x 8 x args.ndf * 4)

    
    with tf.variable_scope("decoder_" + domain_name, reuse=tf.AUTO_REUSE):
        d3 = instance_norm(deconv2d(tf.nn.relu(d2), options.df_dim * 2, name='d3_dconv'), 'd_bn3')
        # d3 is (batch_size x 16 x 16 x args.ndf * 2)
        d4 = instance_norm(deconv2d(tf.nn.relu(d3), options.df_dim, name='d4_dconv'), 'd_bn4')
        # d4 is (batch_size x 32 x 32 x args.ndf)
        d5 = deconv2d(tf.nn.relu(d4), options.output_c_dim, name='d5_dconv')
        # d5 is (batch_size x 64 x 64 x args.output_nc)

    return tf.nn.tanh(d5)
    

def cdann(input):
    
    with tf.variable_scope("cdann", reuse=tf.AUTO_REUSE):
        fg = flip_gradient(input)
        c1 = lrelu(conv2d(fg, 1024, ks=1, s=1, padding='VALID', name='c1_conv'))
        # c1 is (batch_size x 1 x 1 x 1024)
        c2 = lrelu(conv2d(c1, 1024, ks=1, s=1, padding='VALID', name='c2_conv'))
        # c2 is (batch_size x 1 x 1 x 1024)
        c3 = lrelu(conv2d(c2, 1024, ks=1, s=1, padding='VALID', name='c3_conv'))
        # c3 is (batch_size x 1 x 1 x 1024)
        c4 = conv2d(c3, 1, ks=1, s=1, padding='VALID', name='c4_conv')
        # c4 is (batch_size x 1 x 1 x 1)

    return c4

        
def euc_criterion(in_, target):
    return tf.losses.mean_squared_error(target, in_)

    
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))