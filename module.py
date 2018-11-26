from __future__ import division
import tensorflow as tf

from ops import *
from utils import *


def discriminator(image, options, name):
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        
        
def encoder(image, options, domain_name):
    
    with tf.variable_scope("encoder_" + domain_name, reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope("encoder_sharing", reuse=tf.AUTO_REUSE):
        

def decoder(image, options, domain_name):
    
    with tf.variable_scope("decoder_sharing", reuse=tf.AUTO_REUSE):
    
    with tf.variable_scope("decoder_" + domain_name, reuse=tf.AUTO_REUSE):
        
        
def euc_criterion(in_, target):
    return 

    
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))