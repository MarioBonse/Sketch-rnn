# Sketch-rnn
On this project I implemented, with tensorflow 2.0 and keras, Sketch-rnn, a Variational Autoencoder for generating sketches. I followed 
[the original paper](https://arxiv.org/abs/1704.03477 "Sketch-RNN"). 
## Architecture
The architecture is the following. 
![alt text](https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/assets/sketch_rnn_schematic.svg)
## Colab notebook
I have created also a [colab notebook](https://github.com/MarioBonse/Sketch-rnn/blob/master/train_colab_notebook.ipynb) where everyone can train the model and check the results.   
I have trained the model with two different sketches:
* Carrot 
* Cat 
### Some sketch generated!
<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_latent.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_latent.svg" />
</div>
There is also a 
[Jupyter notebook](https://github.com/MarioBonse/Sketch-rnn/blob/master/generator.ipynb)
for testing the sampling. 
We can sample from a hidden variable produced by the encoder or form a sample of a IID gaussian.
The weights of the trained model are in the model directory. 


