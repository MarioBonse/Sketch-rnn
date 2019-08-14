import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import keras.backend as K
from HyperParameters import HP
import math as m

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE():
    def __init__(self):
        self.build_model()  
        # optimizer 
        self.optimizer = tf.keras.optimizers.Adam(lr = HP.lr, clipvalue= HP.grad_clip, decay = HP.lr_decay, epsilon = HP.min_lr)
    
    def build_model(self):
        # build the encoder
        encoderInput = tf.keras.layers.Input(shape = (HP.max_seq_length, HP.input_dimention), batch_size = HP.batch_size, name = "encoder_Input" )
        encoderLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HP.enc_hidden_size, return_sequences=False,
         recurrent_dropout=HP.rec_dropout), merge_mode='concat')(encoderInput)
        self.mu = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(encoderLSTM)
        self.sigma = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(encoderLSTM)
        # latent vaiable z
        z = tf.keras.layers.Lambda(sampling, output_shape=(HP.latent_dim,), name='z')([self.mu, self.sigma])
        # create the model 
        self.encoder = tf.keras.models.Model(encoderInput, [self.mu, self.sigma, z], name='encoder')
        self.encoder.summary()
        #tf.keras.utils.plot_model(self.encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        ' DECODER '
        # now create the decoder
        # input creation
        # sequence as input of the LSTM
        decoder_input_sequence = tf.keras.layers.Input(shape = (HP.max_seq_length, HP.input_dimention), batch_size = HP.batch_size, name = "Decoder_Sequence_Input" )
        # we have also the latent variable as input of the LSTM with the sequence
        inputLatentVariable = tf.keras.layers.RepeatVector(HP.max_seq_length)(z)
        # so we concatentate the two vector
        # as input we have the two vector above concatenated
        totalInput = tf.keras.layers.Concatenate()([decoder_input_sequence, inputLatentVariable])
        

        # Create LSTM for generation with input state = tanh(z)
        decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout, return_sequences=True, return_state=True)
        #
        init_state = tf.keras.layers.Dense(units=(2*decoderLSTM.units), activation='tanh', name = "decoder_init_stat")(z)
        h_0, c_0 = tf.split(init_state, num_or_size_splits=2, axis = 1)
        # creation of the LSTM
        decoder_output, _, _ = decoderLSTM(totalInput, initial_state = [h_0, c_0])

        # dense to output. THe dimention is, as explained in the paper equale to 3 + 
        # 6 times M= number of mixture 
        outputDimention = (3 + HP.M * 6)
        distributionOutput = tf.keras.layers.Dense(outputDimention)(decoder_output)

        # Build Keras model
        self.totalModel = tf.keras.models.Model([encoderInput, decoder_input_sequence], distributionOutput)
        self.totalModel.summary()
        #tf.keras.utils.plot_model(self.totalModel, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    def kl_loss(self, *args, **kwargs):
        # kl loss. 
        kl_value = -0.5*tf.reduce_mean(1+self.sigma-tf.square(self.mu)-tf.exp(self.sigma))
        # the paper introduce a minimum value for the kl loss 
        return tf.maximum(kl_value, HP.KL_min)
    
    def reconstruction_loss(self, y_true, output):
        # reconstruction loss. More complicated loss: 
        # 1. obtain the parameters for the distribution
        q, pi, mux, muy, sigmax, sigmay, ro = self.find_distribution_parameter(output)
        # 2. find posterior for each mixture
        # 2.1 but first find dx, dy, p1, p2, p3
        [dx, dy] = [y_true[:, :, 0], y_true[:, :, 1]]
        penstates = y_true[:,:,2:5]
        mixture_posterior = self.misturegaussian(dx, dy, mux, muy, sigmax, sigmay, ro)
        # now we can obtgain the likelihood of the gaussian
        # first multiply by the Pi terms
        mixture_posterior_weighted = mixture_posterior * pi
        epsilon = tf.constant(1e-6)
        # the loss due t
        L_s = -tf.math.log(tf.math.reduce_sum(mixture_posterior_weighted, 2, keepdims=True) + epsilon )
        #############################################
        # Now the loss due to the pen state. Classical cross entropy
        pklogqk = penstates*tf.math.log(q)
        L_p = -tf.math.reduce_sum(pklogqk, 2, keepdims = True)
        L_r = L_s + L_p
        return L_r

    def total_loss(self):
        kl_loss = self.kl_loss
        self.KL_weight = tf.Variable(HP.wKL, name='kl_weight')
        reconstruction_loss = self.reconstruction_loss
        def my_loss(y_true, y_predicted):
            model_loss = reconstruction_loss(y_true, y_predicted)
            # wheight kl model_loss 
            total_model_loss = self.KL_weight*kl_loss + model_loss
            return total_model_loss
        return my_loss


    def misturegaussian(self, dx, dy, mux, muy, sigmax, sigmay, ro):
        # find N(dx, dy|mux, muy, sigmax, sigmay, ro) for each of the M mixture
        # first of all we have to compute dx - mux. In order to do it we have to 
        # rearrange the dx and dy tensors. We tile the M times, 1 for each mixture
        dx_mux = tf.tile(tf.expand_dims(dx), [1, 1, HP.M]) - mux
        dy_muy = tf.tile(tf.expand_dims(dy), [1, 1, HP.M]) - muy
        Z = tf.square(dx_mux/sigmax) + tf.square(dy_muy/sigmay) + 2.*ro*(dx_mux)*(dy_muy)/sigmax*sigmay
        pi = tf.constant(m.pi)
        denominator = 2.*pi*sigmax*sigmay*tf.math.sqrt(1.-tf.math.square(ro))
        return tf.exp(-Z/(2.*(1. - tf.square(ro))))/denominator
        
    
    def find_distribution_parameter(self, output):
        # the raw output has to be divided into the the distribution parameters.
        # they also have to be normalized
        # 3 parameters for the pen state 
        logit = output[:,:,3]
        # now we have 6 parameter for each gaussian
        distribution_parameter = [output[:, :, (3 + HP.M * (n - 1)):(3 + HP.M * n)] for n in range(1, 7)]
        # now divide them 
        [pi, mux, muy, sigmax, sigmay, ro] = distribution_parameter
        # normalize
        # sigma > 0 -> exp(sigma)
        sigmax = tf. exp(sigmax)
        sigmay = tf. exp(sigmay)
        # -1 < ro < 1 -> tanh
        ro = tf.tanh(ro)
        # logits and pi have to be a convex hull -> softmax
        pi = tf.math.softmax(pi)
        logit = tf.math.softmax(logit)
        return logit, pi, mux, muy, sigmax, sigmay, ro
    
    def compile(self):
        self.totalModel.compile(optimizer=self.optimizer, loss=self.total_loss,
                           metrics=[self.reconstruction_loss, self.kl_loss])



