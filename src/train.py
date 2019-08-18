import data_Manager 
from HyperParameters import HP
import keras
import matplotlib.pyplot as plt
import train_utility as tu
import tensorflow as tf

# callback and data control
# callback that change the weight of the kl loss 
KL_weight = tf.Variable(HP.wKL, name='kl_weight')
KL_wheight_schedule = data_Manager.changing_KL_wheight(KL_weight)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="log")
# import data
datas = data_Manager.Data()
# create the callback for data augmentaion during training
train_generator = data_Manager.DataGenerator(datas)


"""
Create the model
"""
# build the encoder
encoder_input = tf.keras.layers.Input(batch_shape = (HP.batch_size, HP.max_seq_length, HP.input_dimention), name = "encoder_input" )

encoderLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(HP.enc_hidden_size, return_sequences=False,
    recurrent_dropout=HP.rec_dropout), merge_mode='concat')(encoder_input)

hidden_state_mean = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(encoderLSTM)

hidden_state_variance = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(encoderLSTM)

# latent vaiable z 
z = tf.keras.layers.Lambda(tu.sampling, output_shape=(HP.latent_dim,), name='z')([hidden_state_mean, hidden_state_variance])

# create the model with keras
encoder = tf.keras.models.Model(encoder_input, [hidden_state_mean, hidden_state_variance, z], name='encoder')
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

optimizer = tf.keras.optimizers.Adam(lr = HP.lr, clipvalue= HP.grad_clip, 
                decay = HP.lr_decay, epsilon = HP.min_lr)

reconstruction_loss = tu.reconstruction_loss(encoder_input, distribution_output)
kl_loss = tu.kl_loss(hidden_state_mean, hidden_state_variance)

kl_loss *= KL_weight
seq_to_seq_VAE.add_loss(kl_loss)
seq_to_seq_VAE.add_loss(reconstruction_loss)
# compile the model
seq_to_seq_VAE.compile(optimizer=optimizer)
"""
FIT
"""
history = seq_to_seq_VAE.fit_generator(train_generator,steps_per_epoch=(datas.trainDimention)/HP.epochs, 
                                epochs=HP.epochs, callbacks=[KL_wheight_schedule, checkpointer, tensorboard_callback])

#history = seq_to_seq_VAE.fit(datas.train, datas.train , epochs=HP.epochs)



