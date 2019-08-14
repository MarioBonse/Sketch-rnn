import data_Manager
import model 
import tensorflow as tf

# import data
datas = data_Manager.Data()
x_train = datas.x_train
# import the model  
VAEmodel = model.VAE()

# create the callback for data augmentaion during training
datagen = data_Manager.DataGenerator(datas)
model.fit_generator(datagen,
                    steps_per_epoch=len(x_train) / 100, epochs=epochs)



