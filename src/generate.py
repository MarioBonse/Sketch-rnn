import draw
import data_Manager
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from HyperParameters import HP
import train_utility as tu
import random


def sample_from_categorical(pdf):
    x = random.random()
    cum_sum = 0
    for i, p in enumerate(pdf):
        cum_sum += p
        if cum_sum >= x:
            return i
    print("Error dureing sampling\n pdf is ")
    print(pdf)
    return False


def find_distribution_parameter_decoder(output, temperature = 1.):
    # the raw output has to be divided into the the distribution parameters.
    # they also have to be normalized
    # 3 parameters for the pen state 
    logit = output[:,:,:3]
    logit = logit/temperature
    # now we have 6 parameter for each gaussian
    distribution_parameter = [output[:, :, (3 + HP.M * (n - 1)):(3 + HP.M * n)] for n in range(1, 7)]
    # now divide them 
    [pi, mux, muy, sigmax, sigmay, ro] = distribution_parameter
    pi = pi/ temperature
    
    # normalize
    # sigma > 0 -> exp(sigma)
    sigmax = np.exp(sigmax)
    sigmay = np.exp(sigmay)
    # sigma^2 = sigma^2*temperature but we have sigma 
    sigmax = sigmax*np.sqrt(temperature)
    sigmay = sigmay*np.sqrt(temperature)
    # -1 < ro < 1 -> tanh
    ro = np.tanh(ro)
    # logits and pi have to be a convex hull -> softmax
    logit = np.squeeze(logit)
    pi = np.squeeze(pi)
    logit =  np.exp(logit)/sum(np.exp(logit))
    pi = np.exp(pi)/sum(np.exp(pi))
    return logit, pi, np.squeeze(mux), np.squeeze(muy), np.squeeze(sigmax), np.squeeze(sigmay), np.squeeze(ro)

"""
function that generate point by point the sketch
"""
def generate_sketch(decoder, initial_state_net, latent_variable = None, temperature = 1.):
    # Evaluation step (generating text using the learned model)
    starting_point = np.zeros((1,5))
    starting_point[0,2] = 1
    previous_point = starting_point
    if latent_variable == None:
        latent_variable = np.random.rand(1,HP.latent_dim)
    initial_state = initial_state_net.predict(latent_variable)
    h_0, c_0 = tf.split(initial_state, num_or_size_splits=2, axis = 1)
    points = []
    for i in range(HP.max_seq_length):
        new_input = np.concatenate((previous_point, latent_variable), axis =  1)
        new_input =  np.expand_dims(new_input, axis=0)
        predictions = decoder.predict([new_input, h_0, c_0], batch_size = 1, steps = 1)
    
        # obtain mixture parameters
        LSTM_output = predictions[0]
        
        # obtain states h_new and c_new
        h_0 = predictions[1]
        c_0 = predictions[2]
        
        
        
        q, pi, mux, muy, sigmax, sigmay, ro = find_distribution_parameter_decoder(LSTM_output, temperature = temperature)     
        
        print(q)
        # sample from the categorical distribution
        q_index = sample_from_categorical(q)
        if q_index == 2:
            print("end of sketch")
            points.append(np.array([0,0,1,0,0]))
            return np.array(points)
        
        pi_index = sample_from_categorical(pi)
        
        # sample from the bivariate normals
        new_sigma_x = sigmax[pi_index]
        new_sigma_y = sigmay[pi_index]
        new_ro = ro[pi_index]
        new_mu_x = mux[pi_index]
        new_mu_y = muy[pi_index]
        covariance_matrix = [[new_sigma_x**2, new_ro*new_sigma_x*new_sigma_y],  [ new_ro*new_sigma_x*new_sigma_y, new_sigma_y]]
        dx, dy = np.random.multivariate_normal(mean = (new_mu_x, new_mu_y), cov = covariance_matrix)
        new_elem = [dx, dy, 0, 0, 0]
        new_elem[2+q_index] = 1
        new_elem =np.array(new_elem)
        points.append(new_elem)
        previous_point = np.expand_dims(new_elem, axis = 0)
    return np.array(points)


if __name__ == "__main__":
    # model for predicting the inital state 
    batch_z = tf.keras.Input(shape=(HP.latent_dim,))
    initial_state = tf.keras.layers.Dense(units=(2*HP.dec_hidden_size), activation='tanh', name = "decoder_init_stat")(batch_z)
    latent_to_hidden_state_model = tf.keras.Model(inputs=batch_z, outputs=initial_state)
    latent_to_hidden_state_model.load_weights("model_weight.h5", by_name = True)
    latent_to_hidden_state_model.summary()

    # create the LSTM for generating
    """
    We have 3 input tensor. The input of the LSTM and the hidden states 
    """
    decoder_input = tf.keras.Input(shape=(1, 5 + HP.latent_dim))
    initial_h_input = tf.keras.Input(shape=(HP.dec_hidden_size,))
    initial_c_input = tf.keras.Input(shape=(HP.dec_hidden_size,))
    # now the LSTM
    decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout, 
                                        return_sequences=True, return_state=True, name = "LSTM_decoder")

    # creation of the LSTM
    decoder_output, h_new, c_new = decoderLSTM(decoder_input, initial_state = [initial_h_input, initial_c_input])
    # dense to output. THe dimention is, as explained in the paper equal to 3 + 6*M
    # 6 times M= number of mixture 
    output_dimention = (3 + HP.M * 6)
    distribution_output = tf.keras.layers.Dense(output_dimention, name = "output_layer")(decoder_output)

    # Now we load the weights from the trained model
    generator = tf.keras.models.Model([decoder_input, initial_h_input, initial_c_input], outputs =[ distribution_output , h_new, c_new])
    generator.summary()
    generator.load_weights("model_weight.h5", by_name = True)
    generator.build(tf.TensorShape([1, None])) 

    seq =  generate_sketch(generator, latent_to_hidden_state_model)

