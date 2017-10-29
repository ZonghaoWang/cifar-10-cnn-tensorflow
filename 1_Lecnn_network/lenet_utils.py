# -*- coding: utf-8 -*-
import tensorflow as tf
# %%
def weight_variable(shape, name="weight_variable", init_fun=tf.random_normal):
    '''Helper function to create a weight variable initialized with
    a normal distribution

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    with tf.name_scope(name):
        initial = init_fun(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)

# %%
def bias_variable(shape, name="bias_variable", init_fun=tf.random_normal):
    '''Helper function to create a bias variable initialized with
    a constant value.

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    with tf.name_scope(name):
        initial = init_fun(shape, mean=0.0, stddev=0.01)
        return tf.Variable(initial)