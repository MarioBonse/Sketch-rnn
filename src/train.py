import data_Manager 
from HyperParameters import HP
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import math as m
"""
Utility functions
"""

"""
Function for the reparametrization trick
"""
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon

"""
loss due to the KL Divergence
"""
def kl_loss(hidden_state_mean, hidden_state_variance):

    kl_value = tf.math.reduce_sum(1+hidden_state_variance-tf.math.square(hidden_state_mean)-tf.math.exp(hidden_state_variance), axis = 1)  
    kl_value = -kl_value/2.*HP.latent_dim 
    # the paper introduce a minimum value for the kl loss 
    return tf.math.maximum(kl_value, HP.KL_min)

"""
RECONSTRUCTION loss
"""
def reconstruction_loss(y_true, output): 
    # divide the output into the different parameters
    q, pi, mux, muy, sigmax, sigmay, ro = divide_oputput(output)


    # 2.1 but first find dx, dy, p1, p2, p3 
    [dx, dy] = [y_true[:, :, 0], y_true[:, :, 1]]
    penstates = y_true[:,:,2:]

    mixture_posterior = bivariate_normal_pdf(dx, dy, mux, muy, sigmax, sigmay, ro)
    # now we can calculate the likelihood 
    # first multiply by the Pi terms
    mixture_posterior_weighted = mixture_posterior*pi
    epsilon = tf.constant(1e-6)

    L_s = tf.math.log(tf.math.reduce_sum(mixture_posterior_weighted, 2, keepdims=True) + epsilon )
    # create a vector equal to zero where the strokes end -> for masking displacement where strokes ends
    mask = tf.expand_dims(1. - penstates[:,:,2], -1)

    L_s = L_s * mask
    L_s = -tf.reduce_sum(L_s, axis = 1)
    

    # loss due to the pen state -> cross entropy
    L_p = tf.math.reduce_sum(penstates*tf.math.log(q + epsilon), axis = 2, keepdims=True)
    L_p = -tf.math.reduce_sum(L_p, axis = 1)
    L_r = L_s + L_p
    L_r = L_r/HP.max_seq_length
    return L_r



def bivariate_normal_pdf(dx, dy, mux, muy, sigmax, sigmay, ro):
    """
     find N(dx, dy|mux, muy, sigmax, sigmay, ro) for each of the M mixture
     ref https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """
    # first of all we have to compute dx - mux. In order to do it we have to 
    # rearrange the dx and dy tensors. We tile them M times, 1 for each \Pi
    dx_mux = tf.tile(tf.expand_dims(dx, -1), [1, 1, HP.M]) - mux
    dy_muy = tf.tile(tf.expand_dims(dy, -1), [1, 1, HP.M]) - muy
    sigmax_times_sigmay = sigmax*sigmay

    #_ = K.print_tensor(tf.shape(dx_mux), message='K.min(dx_mux) = ')
    
    Z = tf.square(dx_mux/sigmax) + tf.square(dy_muy/sigmay) - (2.*(ro*(dx_mux)*(dy_muy)))/sigmax_times_sigmay
    one_minus_ro = (1. - tf.square(ro))
    log_mixture = -Z/(2.*one_minus_ro)

    # denominator
    pi_times_t = 2*tf.constant(m.pi)
    denominator = pi_times_t*(sigmax_times_sigmay*tf.math.sqrt(one_minus_ro))
    mixture_final = tf.math.exp(log_mixture)/denominator

    return mixture_final   
    
def divide_oputput(output):
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
    sigmax = tf.math.exp(sigmax)
    sigmay = tf.math.exp(sigmay)
    # -1 < ro < 1 -> tanh
    ro = tf.math.tanh(ro)
    # logits and pi have to be a convex hull -> softmax
    pi = tf.math.softmax(pi)
    q = tf.math.softmax(logit)
    return q, pi, mux, muy, sigmax, sigmay, ro


if __name__ == "__main__":
    """
    Create the model

    1. ENCODER
    """
    encoder_input = tf.keras.layers.Input(batch_shape = (HP.batch_size, None, HP.input_dimention), name = "encoder_input" )

    encoderLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HP.enc_hidden_size, return_sequences=False,
        recurrent_dropout=HP.rec_dropout, name = "LSTM_encoder"), merge_mode='concat', name = "BI_LSTM_encoder")(encoder_input)

    hidden_state_mean = tf.keras.layers.Dense(HP.latent_dim, activation='linear', name = "mean_MLP")(encoderLSTM)

    variance_hat = tf.keras.layers.Dense(HP.latent_dim, activation='linear', name = "variance_MLP")(encoderLSTM)

    # from mean and variance to latent vairable z
    z = tf.keras.layers.Lambda(sampling, output_shape=(HP.latent_dim,), name='z')([hidden_state_mean, variance_hat])

    # create the model with keras
    encoder = tf.keras.models.Model(encoder_input, [hidden_state_mean, variance_hat, z], name='encoder')
    encoder.summary()
    #tf.keras.utils.plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    """ 
    DECODER 
    """
    # the decoder LSTM has an input composed by the sequence and also the hidden state
    decoder_input_sequence = tf.keras.layers.Input(batch_shape = (HP.batch_size, HP.max_seq_length, HP.input_dimention), name = "decoder_seq_input" )
    # we have also the latent variable as input of the LSTM with the sequence
    inputLatentVariable = tf.keras.layers.RepeatVector(HP.max_seq_length)(z)
    # so we concatentate the two vector
    # as input we have the two vector above concatenated
    totalInput = tf.keras.layers.Concatenate()([decoder_input_sequence, inputLatentVariable])
    # Create LSTM for generation with input state = tanh(z)
    decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout, 
                                        return_sequences=True, return_state=True, name = "LSTM_decoder")
    #
    init_state = tf.keras.layers.Dense(units=(2*decoderLSTM.units), activation='tanh', name = "decoder_init_stat")(z)
    h_0, c_0 = tf.split(init_state, num_or_size_splits=2, axis = 1)

    # creation of the LSTM
    decoder_output, _, _ = decoderLSTM(totalInput, initial_state = [h_0, c_0])

    # dense to output. THe dimention is, as explained in the paper equal to 3 + 6*M
    # 6 times M= number of mixture 
    output_dimention = (3 + HP.M * 6)
    distribution_output = tf.keras.layers.Dense(output_dimention, name = "output_layer")(decoder_output)

    # Build Keras model
    seq_to_seq_VAE = tf.keras.models.Model([encoder_input, decoder_input_sequence], distribution_output)
    seq_to_seq_VAE.summary()
    tf.keras.utils.plot_model(seq_to_seq_VAE, to_file='vae.png', show_shapes=True)

    optimizer = tf.keras.optimizers.Adam(lr = HP.lr, clipvalue= HP.grad_clip)



    """

    recon_loss = reconstruction_loss(encoder_input, distribution_output)
    kl_los = kl_loss(hidden_state_mean, variance_hat)

    kl_loss *= KL_weight
    #seq_to_seq_VAE.add_loss(kl_loss)
    #seq_to_seq_VAE.add_loss(reconstruction_loss)
    # compile the model
    """

    # variable for the weight of kl divergece
    #if I use tf2:0 -> KL_weight = tf.Variable(HP.wKL, name='kl_weight')

    KL_weight = tf.keras.backend.variable(0.01, name = 'kl_weight')
    KL_wheight_schedule = data_Manager.changing_KL_wheight(KL_weight)
    def vae_loss(y_true, y_pred):
        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
        # E[log P(X|z)]
        recon = reconstruction_loss(y_true, y_pred)
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl = kl_loss(hidden_state_mean, variance_hat)

        return recon + kl*KL_weight

    # callback and data control
    # callback that change the weight of the kl loss 


    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log")
    # import data
    datas = data_Manager.Data()
    # create the callback for data augmentaion during training
    train_generator = data_Manager.DataGenerator(datas.train)
    validation_encoder = datas.valid
    validation_decoder = data_Manager.create_decoder_input(validation_encoder)
    vaidation = [validation_encoder, validation_decoder]


    seq_to_seq_VAE.compile(optimizer=optimizer, loss = vae_loss)
    """
    FIT
    """


    history = seq_to_seq_VAE.fit_generator(train_generator,
                validation_data= ([validation_encoder, validation_decoder],[validation_encoder]),
                            steps_per_epoch=(datas.trainDimention)/HP.batch_size, 
                            epochs=HP.epochs, callbacks=[tensorboard_callback])
    # save the model
    seq_to_seq_VAE.save_weights("model_weight.h5")
    seq_to_seq_VAE.save("model.h5")



