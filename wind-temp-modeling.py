import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
from matplotlib import pyplot as plt

class Model:
    def __init__(self, percent_train_test = 0.8, percent_train_val = 0.8, window_size = 10, activation = "tanh",loss_fxn="MSE", 
    epochs = 70, batch_size = 15, verbose = 1, shuffle = True ):
        self.train_amount = percent_train_test
        self.percent_train_val = percent_train_val
        self.num_time = window_size
        self.activation = activation
        self.loss_fn = loss_fxn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle



    def load_data(self):
        # read in data
        data = Dataset('./air.mon.mean.nc', 'r')
        T = data.variables['air'][:,0]
        t = data.variables['time'][:]
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        data.close()

        data = Dataset('./uwnd.mon.mean.nc', 'r')
        u = data.variables['uwnd'][:,0]
        data.close()

        data = Dataset('./vwnd.mon.mean.nc', 'r')
        v = data.variables['vwnd'][:,0]
        data.close()
        
        T = np.flip(T, axis=1)
        u = np.flip(u, axis=1)
        v = np.flip(v, axis=1)

        # extract data of the US
        lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
        lon_us = np.argwhere(np.logical_and(lon>-123.3+180, lon<-81.2+180))[:,0]
        lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

        T_US = T[:, lat_ind, lon_ind]
        u_US = u[:, lat_ind, lon_ind]
        v_US = v[:, lat_ind, lon_ind]

        # combine u, v, T into one input tensor of shape (num_samples, num_t, nlat, nlon, 3)
        X = np.zeros((u_US.shape[0], u_US.shape[1], u_US.shape[2], 3))
        print('input shape:', X.shape)

        X[:,:,:,0] = T_US
        X[:,:,:,1] = u_US
        X[:,:,:,2] = v_US
        X = X.reshape(int(X.shape[0]/self.num_time), self.num_time, X.shape[1],X.shape[2],X.shape[3])
        # partition train and test data
        #  80% train, 20% test
        num_train = int(X.shape[0]*self.train_amount)
        X_train_old = X[:num_train]
        Y_train_old = X[1:num_train + 1]
        X_test = X[num_train:-1]  # reserve the last time step as the label
        # shift train or test data by one time step to get the labels
        Y_test = X[num_train+1:]
        # further divide train data into train and validation data
        num_train_sub = int(num_train*self.percent_train_val)
        
        X_train = X_train_old[:num_train_sub]
        Y_train = X_train_old[1 : num_train_sub+1]
        
        X_val = X_train_old[num_train_sub : -1]
        Y_val = X_train_old[num_train_sub+1:]
        self.X_train_plus_val = X_train_old
        self.Y_train_plus_val = Y_train_old
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.channels = X.shape[4]
        self.rows  = X.shape[2]
        self.cols = X.shape[3]
        print('x train shape:', X_train.shape)
        print('y train shape:', Y_train.shape)
        print('x val shape:', X_val.shape)
        print('y val shape:', Y_val.shape)
        print('x test shape:', X_test.shape)
        print('y test shape:', Y_test.shape)


        return {"X_train": X_train, "Y_train": Y_train, \
                "X_val":X_val, "Y_val": Y_val,\
                "X_test":X_test, "Y_test":Y_test}

    def model_setup(self):
        # TO DO: Make this more accessible so layers can be defined in an easier manner
        model = keras.Sequential(
            [
                keras.layers.Input(
                    shape=(self.num_time, self.rows, self.cols,self.channels), name="input_layer"
                ),
                keras.layers.ConvLSTM2D(
                    filters = 40, kernel_size = (3, 3), return_sequences = True, data_format = "channels_last", input_shape = (self.num_time, self.rows, self.cols,self.channels), padding="same", activation = self.activation
                ),               
                keras.layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
                ),
                layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
                ),
                layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
                ),
                layers.BatchNormalization(),
                layers.Conv3D(
                    filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )
        model.compile(loss = self.loss_fn,optimizer="adam")
        self.model = model
        print(model)
        return

    def training(self):
        self.model.fit(
            self.X_train_plus_val, self.Y_train_plus_val, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
            validation_split=self.percent_train_val, shuffle = self.shuffle
        )
        return
    
def main():
    convLstm = Model()
    convLstm.load_data()
    convLstm.model_setup()
    convLstm.training()

if __name__ == "__main__":
    main()