import urllib.request
import glob
import os
import numpy as np
from HyperParameters import HP
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import Callback


class Data():
    # managing the data
    def __init__(self, size = None):
        'Initialization'
        # first load the data
        self.loadData(size)
        self.dim = HP.input_dimention
        self.batch_size = HP.batch_size
        self.train = purify(self.train)
        self.valid = purify(self.valid)
        self.test = purify(self.test)
        self.train = self.normalize(self.train)
        self.valid = self.normalize(self.valid)
        self.test = self.normalize(self.test)
        self.train = np.array(self.train)
        self.valid = np.array(self.valid)
        self.test = np.array(self.test)
    

    def loadData(self, size):  
        try:
            npzFile = np.load(HP.data_location, allow_pickle=True, encoding='latin1')
        except:
            npzFile = np.load("../"+HP.data_location, allow_pickle=True, encoding='latin1')
        train = npzFile['train']
        if size:
            self.train = np.copy(train[:size])
        else:
            self.train = train
        self.trainDimention = len(self.train)
        
        self.test = npzFile['test']
        
        self.valid = npzFile['valid']
        self.validationDimention = len(self.valid)
        return self.train, self.valid, self.test

    def calculate_normalizing_scale_factor(self, strokes):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(strokes)):
            if len(strokes[i]) > HP.max_seq_length:
                continue
            for j in range(len(strokes[i])):
                data.append(strokes[i][j, 0])
                data.append(strokes[i][j, 1])
        data = np.array(data)
        return np.std(data)

    def normalize(self, strokes,  scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor(strokes)
        self.scale_factor = scale_factor
        for i in range(len(strokes)):
            strokes[i][:, 0:2] /= self.scale_factor
        return strokes

def purify(strokes):
    # We have to remove too long sequence 
    data = []
    for seq in strokes:
        if seq.shape[0] <= HP.max_seq_length:
            len_seq = len(seq[:,0])
            # pen state made by 3 state
            new_seq = np.zeros((HP.max_seq_length,5))
            new_seq[:len_seq,:2] = seq[:,:2]
            new_seq[:len_seq-1,2] = 1-seq[:-1,2]
            new_seq[:len_seq,3] = seq[:,2]
            new_seq[len_seq:,4] = 1
            data.append(new_seq)
    return data



def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result

"""    # Normalize input Dx, Dy. We only remove the std as explained in the paper
    def calculate_normalizing_scale_factor(self, strokes):
      data = []
      for element in strokes:
          for point in element:
              data.append(point[0])
              data.append(point[1])
      return np.std(np.array(data))

    def normalize(self, strokes):
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(strokes)
        for seq in strokes:
            seq[:, 0:1] /= scale_factor
            data.append(seq)
        return data
"""



# see https://keras.io/utils/ for more info
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, Data, shuffle=True, validation = False):
        'Initialization'
        self.Data = Data
        self.validation = validation
        if validation:
            self.batch_size = len(Data)
        else:
            self.batch_size = HP.batch_size
        self.dimention = len(Data)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.validation:
            return 1    
        return int(np.floor(self.dimention/ self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        encoder_input = self.Data[indexes]
        #if not self.validation:
            #encoder_input = self.dataAugmentation(encoder_input)
        decoder_ipnut = create_decoder_input(encoder_input)
        """
        encoder_ipnut_short = []
        for i in range(encoder_input.shape[0]):
            for j in range(encoder_input.shape[1]):
                if encoder_input[i,j,4] == 1:
                    encoder_ipnut_short.append(np.copy(encoder_input[i, :j, :]))
                    break
        """
        
        return [encoder_input, decoder_ipnut], [encoder_input]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.dimention)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def dataAugmentation(self, strokes):
        # generate random uniform between 0.9 to 1.1
        randomx = np.random.rand()*(0.1)+1.
        randomy = np.random.rand()*(0.1)+1.
        # multiply the 
        strokes[:,:,0] = strokes[:,:,0]*randomx
        strokes[:,:,1] = strokes[:,:,1]*randomy
        return strokes     

def create_decoder_input(sequence):
    """
    function that, given an input sequence returns another sequence 
    for the decoder. 
    It shift the old sequence by one and insert on head the value
    (0,0,1,0,0)
    """
    decoder_ipnut = np.zeros(shape=sequence.shape)
    # copy the value of the sequence
    decoder_ipnut[:,1:] = sequence[:,:-1]
    decoder_ipnut[:,0] = np.array([0,0,1,0,0])

    return decoder_ipnut
        

class changing_KL_wheight(Callback):
    def __init__(self, kl_weight, verbose = 1, mu_min = 0.01):
        super(Callback, self).__init__()
        self.kl_wheight = kl_weight
        self.verbose = verbose
        self.curr_mu = 0

    def on_epoch_begin(self, epochs, logs = {}):
        self.curr_mu = 1 - (1-HP.eta_min)*HP.R**epochs
        New_wheight_kl = (self.curr_mu)*HP.wKL
        # IF I USE TF-2.0 then I have to update the variable like that
        self.kl_wheight.assign(New_wheight_kl)
        #tf.keras.backend.set_value(self.kl_wheight, New_wheight_kl)


    def on_train_batch_begin(self, epochs, logs = {}):
        pass

    def on_train_batch_end(self, epochs, logs = {}):
        pass

    def on_test_batch_begin(self, epochs, logs = {}):
        pass

    def on_test_batch_end(self, epochs, logs = {}):
        pass
    
    def on_test_begin(self, *arg, **karg):
        pass

    def on_test_end(self, *arg, **karg):
        pass
