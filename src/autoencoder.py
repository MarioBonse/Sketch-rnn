from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from .data import load_data, normalize
from .HyperParameters import HP
from .BRNNEncoder import BidirectionalLSTM

from keras.layers import Dense, Input
from keras.layers import LSTM, Lambda
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.layers.merge import Concatenate
from keras.layer import Model



from keras.layers.core import RepeatVector

class VAE():
    def __init__(self, encoder, input_size):# load dataset
        # VAE model = encoder + decoder
        # build encoder model 
        # Bidirectional neural network. 
        self.enocder_obj = encoder(input_size, HP.enc_hidden_size)
        inputs = self.encoder_onbj.inputs
        #inputs = self.encoder.inputs
        #self.encoder_net = Dense(2*HP.latent_dim, activation='linear')(self.enocder_obj.model)
        mu = Dense(HP.latent_dim, activation='linear')(self.enocder_obj.model)
        sigma = Dense(HP.latent_dim, activation='linear')(self.enocder_obj.model)
        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(HP.latent_dim,), name='z')([mu, sigma])

        # instantiate encoder model
        self.encoder = Model(inputs, [mu, sigma, z], name='encoder')
        self.encoder.summary()

        """ Now we build the decoder
        # build decoder model
        # latent hidden variable is fed as input for each preiction 
        """
        z_decoder_input = RepeatVector(HP.max_seq_length)(z)
        # And also the oputput of the previus state after the GMM
        previus_output_decoder_input = Input(shape=(input_size, HP.max_seq_length), name='RNN_Decoder')
        full_input = Concatenate()([previus_output_decoder_input, z_decoder_input])
        decoderLSTM = LSTM(units=HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout,
                                            return_sequences=True, return_state=True)(full_input)
        # as explained in the paper the output dimention is equal to 5M + M + 3 with M the number of 
        # mixture gaussian we decide to use. M is an Hyper parameter
        output_dimention = 6*HP.M + 3
        decoder_net =  Dense(output_dimention, activation='linear')(decoderLSTM)


        # instantiate decoder model
        self.decoder = Model(full_input, decoder_net, name='decoder')
        self.decoder.summary()


    def sample(self, eps=None):
        if eps is None:
            eps = tf.random_normal(shape=(100, HP.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        result = self.encoder_net(x)
        mean = result[:HP.latent_dim]
        logvar = result[HP.latent_dim:]
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
    #encode
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

optimizer = tf.train.AdamOptimizer(1e-4)
def apply_gradients(optimizer, gradients, variables, global_step=None):
  optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


if __name__ == '__main__':
    train, validation, test = load_data()
    train = normalize(train)
    # TODO verify if I have to change the 3 parameters. probably yes
    input_size = train[0][0].size


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
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.mean(reconstruction_loss + kl_loss)
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