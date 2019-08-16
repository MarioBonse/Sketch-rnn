import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import keras.backend as K
from HyperParameters import HP
from keras.losses import categorical_crossentropy
import math as m

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

    
def kl_loss(hidden_state_mean, hidden_state_variance):
    # kl loss. 
    kl_value = -0.5*K.mean(1+hidden_state_variance-K.square(hidden_state_mean)-K.exp(hidden_state_variance))
    # the paper introduce a minimum value for the kl loss 
    return K.maximum(kl_value, HP.KL_min)
    
def reconstruction_loss(y_true, output):
    # reconstruction loss. More complicated loss: 
    # 1. obtain the parameters for the distribution
    q, pi, mux, muy, sigmax, sigmay, ro = find_distribution_parameter(output)
    # 2. find posterior for each mixture
    # 2.1 but first find dx, dy, p1, p2, p3 
    [dx, dy] = [y_true[:, :, 0], y_true[:, :, 1]]
    # we have to stack the value for each 
    penstates = y_true[:,:,2:5]
    mixture_posterior = misturegaussian(dx, dy, mux, muy, sigmax, sigmay, ro)
    # now we can obtgain the likelihood of the gaussian
    # first multiply by the Pi terms
    mixture_posterior_weighted = mixture_posterior * pi
    epsilon = K.constant(1e-6)
    # the loss due the dx, dy 
    # [100, 200, 1]
    L_s = -K.log(K.sum(mixture_posterior_weighted, 2, keepdims=True) + epsilon )

    #############################################
    # Now the loss due to the pen state. Classical cross entropy
    L_p = categorical_crossentropy(penstates, q)
    L_p = K.expand_dims(L_p)
    L_r = L_s + L_p
    return L_r


def misturegaussian(dx, dy, mux, muy, sigmax, sigmay, ro):
    # find N(dx, dy|mux, muy, sigmax, sigmay, ro) for each of the M mixture
    # first of all we have to compute dx - mux. In order to do it we have to 
    # rearrange the dx and dy tensors. We tile the M times, 1 for each mixture
    dx_mux = K.tile(K.expand_dims(dx, -1), [1, 1, HP.M]) - mux
    dy_muy = K.tile(K.expand_dims(dy, -1), [1, 1, HP.M]) - muy
    Z = K.square(dx_mux/sigmax) + K.square(dy_muy/sigmay) + 2.*ro*(dx_mux)*(dy_muy)/sigmax*sigmay
    pi = K.constant(m.pi)
    denominator = 2.*pi*sigmax*sigmay*K.sqrt(1.-K.square(ro))
    return K.exp(-Z/(2.*(1. - K.square(ro))))/denominator
        
    
def find_distribution_parameter(output):
    # the raw output has to be divided into the the distribution parameters.
    # they also have to be normalized
    # 3 parameters for the pen state 
    logit = output[:,:,:3]
    # now we have 6 parameter for each gaussian
    distribution_parameter = [output[:, :, (3 + HP.M * (n - 1)):(3 + HP.M * n)] for n in range(1, 7)]
    # now divide them 
    [pi, mux, muy, sigmax, sigmay, ro] = distribution_parameter
    # normalize
    # sigma > 0 -> exp(sigma)
    sigmax = K.exp(sigmax)
    sigmay = K.exp(sigmay)
    # -1 < ro < 1 -> tanh
    ro = K.tanh(ro)
    # logits and pi have to be a convex hull -> softmax
    pi = K.softmax(pi)
    logit = K.softmax(logit)
    return logit, pi, mux, muy, sigmax, sigmay, ro
    


