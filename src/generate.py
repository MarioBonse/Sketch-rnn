import tensorflow as tf
import numpy as np
from HyperParameters import HP
import train_utility as tu
"""
we create only the decoder in order to generate data
"""
# the decoder LSTM has an input composed by the sequence and also the hidden state
decoder_input_sequence = tf.keras.layers.Input(shape = (HP.max_seq_length, HP.input_dimention), name = "decoder_seq_input" )
# we have also the latent variable as input of the LSTM with the sequence
latent_variable = tf.keras.layers.Input(shape = (HP.latent_dim), name = "latent_variable" )
inputLatentVariable = tf.keras.layers.RepeatVector(HP.max_seq_length)(latent_variable)
# so we concatentate the two vector
# as input we have the two vector above concatenated
totalInput = tf.keras.layers.Concatenate()([decoder_input_sequence, inputLatentVariable])
# Create LSTM for generation with input state = tanh(z)
decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, recurrent_dropout=HP.rec_dropout, 
                                    return_sequences=True, return_state=True, name = "LSTM_decoder")
#
init_state = tf.keras.layers.Dense(units=(2*decoderLSTM.units), activation='tanh', name = "decoder_init_stat")(latent_variable)
h_0, c_0 = tf.split(init_state, num_or_size_splits=2, axis = 1)
# creation of the LSTM
decoder_output, _, _ = decoderLSTM(totalInput, initial_state = [h_0, c_0])

# dense to output. THe dimention is, as explained in the paper equal to 3 + 6*M
# 6 times M= number of mixture 
output_dimention = (3 + HP.M * 6)
distribution_output = tf.keras.layers.Dense(output_dimention, name = "output_layer")(decoder_output)

# Now we load the weights from the trained model
generator = tf.keras.models.Model([decoder_input_sequence, latent_variable], distribution_output)
generator.summary()
generator.load_weights(HP.model_folder+ HP.model_name, by_name = True)
generator.build(tf.TensorShape([1, None])) 
generator.summary/()

"""
funciton that generate point by point the sketch
"""
def generate_text(model, latent_varoable, temperature = 1):
    # Evaluation step (generating text using the learned model)
    points_generated = []
    starting_point = np.array([0,0,1,0,0])
    previous_point = starting_point

    model.reset_states()
    for i in range(HP.max_seq_length):
        new_input = tf.concatenate(previus_point, latent_variable)
        predictions = model(new_input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        q, pi, mux, muy, sigmax, sigmay, ro = tu.find_distribution_parameter(predictions, temperature = temperature)
        
        # sample from the bivariate normals
        covariance_matrix = np.matrix([sigmax**2, ro*sigmax*sigmay], [ ro*sigmax*sigmay, sigmay**2])
        newdelta = np.random.multivariate_normal(mean = np.array(mux, muy), cov = covariance_matrix)
        

        # using the GMM to predict dx, dy and categorical to predice p_i
        

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))