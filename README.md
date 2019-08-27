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

#### Carrot where the hidden variable is sampled from the IID gaussian:
<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_IID_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_IID_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_IID_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_IID_4.svg" />
</div>

#### Carrot where the latent variable is encoded from a sketch

<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_sketch_latent_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_sketch_latent_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_sketch_latent_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_from_sketch_latent_4.svg" />
</div>

#### Cat where the hidden variable is sampled from the IID gaussian:
<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_IID_gaussian_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_IID_gaussian_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_IID_gaussian_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_IID_gaussian_4.svg" />
</div>

#### Cat where the latent variable is encoded from a sketch

<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_sketch_latent_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_sketch_latent_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_sketch_latent_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_from_sketch_latent_4.svg" />
</div>

#### Example of skatches from the *Quick, draw!* dataset:


<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_original_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_original_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_original_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/carrot_original_4.svg" />
</div>

<div align="left">
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_original_1.svg" />
<img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_original_2.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_original_3.svg" />
  <img src="https://github.com/MarioBonse/Sketch-rnn/blob/master/results/cat_original_4.svg" />
</div>

There is also a 
Jupyter notebook
for testing the sampling. 
We can sample from a hidden variable produced by the encoder or form a sample of a IID gaussian.
The weights of the trained model are in the model directory. 


