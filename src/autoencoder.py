from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import LSTM, Lambda
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from .data import load_data, normalize
from .HyperParameters import HP
from .BRNNEncoder import BidirectionalLSTM

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
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


# load dataset
train, validation, test = load_data()
train = normalize(train)
# TODO verify if I have to change the 3 parameters. probably yes
input_size = train[0][0].size
# VAE model = encoder + decoder
# build encoder model 
# Bidirectional neural network. 
BDRNNencoder = BidirectionalLSTM(input_size, HP.enc_hidden_size)
inputs = BDRNNencoder.inputs
mu = Dense(HP.latent_dim, activation='linear')(BDRNNencoder.model)
sigma = Dense(HP.latent_dim, activation='linear')(BDRNNencoder.model)
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(HP.latent_dim,), name='z')([mu, sigma])

# instantiate encoder model
encoder = Model(inputs, [mu, sigma, z], name='encoder')
encoder.summary()

# Now we build the decoder
# build decoder model
decoder_input = Input(shape=(input_size, HP.max_seq_length), name='RNN_Decoder')
decoderLSTM = LSTM(units=HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout,
                                      return_sequences=True, return_state=True)(decoder_input)


# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_mlp_mnist.h5')

    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")