
class hyper_parameters():
    def __init__(self):
        # model location fodler
        self.model_folder = "model/"
        self.model_name = "sketch-RNN.h5"
        # Location of the data and name of the file
        # name = sketchrnn_airplane.full.npz
        self.data_folder = "data/"
        self.data_name = "cat.npz"
        self.data_location = self.data_folder+self.data_name
        # NN parameters
        self.latent_dim = 256 
        self.input_dimention = 5
        self.enc_hidden_size = 256
        self.dec_hidden_size = 512 
        self.Nz = 128
        self.M = 20
        self.rec_dropout = 0.1
        self.batch_size = 100
        self.eta_min = 0.01
        self.R = 0.99995
        self.KL_min = 0.2
        self.wKL = 0.5
        self.lr = 0.001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 1.
        self.temperature = 0.4
        self.max_seq_length = 200
        self.epochs = 100


HP = hyper_parameters()