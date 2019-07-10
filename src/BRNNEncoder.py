# object for managing the encoder as BRNN. 
from .HyperParameters import HP
from keras import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Input

class BidirectionalLSTM():
    def __init__(self, input_dimention, output_dimention):
        inputs = Input(shape=input_dimention, name='encoder_input')
        self.model = Bidirectional(LSTM(HP.enc_hidden_size, return_sequences=False, 
        recurrent_dropout=HP.rec_dropout), merge_mode='concat')(inputs)
        self.inputs = inputs



