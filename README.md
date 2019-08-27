# Sketch-rnn
On this project I implemented Sketch-rnn, a Variational Autoencoder for generating sketches. I followed 
[the original paper](https://arxiv.org/abs/1704.03477 "Sketch-RNN"). 
## Architecture
The architecture is the following. 
![alt text](https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/assets/sketch_rnn_schematic.svg)
## Colab notebook
I have created also a [colab notebook](https://github.com/MarioBonse/Sketch-rnn/blob/master/train_colab_notebook.ipynb) where everyone can train the model and check the results.   
I have trained the model with two different sketches:
* Carrot ![alt text](https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_latent.svg)
* Cat ![alt text](https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_latent.svg)
Both can be campled from a IID, if the latent variable argoument is not passed to the function,
or from an hidden variabe Z encoded by the encoder.
There is also a Jupyter notebook for testing the sampling. 
The weights of the trained model are in the [model](https://github.com/MarioBonse/Sketch-rnn/blob/master/train_colab_notebook.ipynb) directory. 


