import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

USE_GPU = False

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

class Model:
    def __init__(self, percent_train_test = 0.8, percent_train_val = 0.8, window_size = 24, activation = "relu",loss_fxn="MSE", 
    epochs = 50, batch_size = 24, verbose = 1, shuffle = True ):
        self.train_amount = percent_train_test
        self.percent_train_val = percent_train_val
        self.num_time = window_size
        self.activation = activation
        self.loss_fn = loss_fxn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.inputs=  []
        self.targets =  []
        self.outputs =  []
        #self.scaler_dict = {}
        #self.scaler = "Standard"


    def load_data(self):
        # read in data
        dir = './'
        data = Dataset(dir + 'air.mon.mean.nc', 'r')
        T = data.variables['air'][:,0]
        t = data.variables['time'][:]
        lat = data.variables['lat'][:]
        lon = data.variables['lon'][:]
        data.close()

        data = Dataset(dir + 'uwnd.mon.mean.nc', 'r')
        u = data.variables['uwnd'][:,0]
        data.close()

        data = Dataset(dir + './vwnd.mon.mean.nc', 'r')
        v = data.variables['vwnd'][:,0]
        data.close()
        
        T = np.flip(T, axis=1)
        u = np.flip(u, axis=1)
        v = np.flip(v, axis=1)

        # extract data of the US
        lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
        lon_us = np.argwhere(np.logical_and(lon > np.mod(-123.3,360),lon < np.mod(-81.2, 360)))[:,0]
        lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

        n_sample = int(T.shape[0]/self.num_time)*self.num_time
        print('******num samp=', n_sample,'*******')

        T_US = T[:n_sample, lat_ind, lon_ind]
        u_US = u[:n_sample, lat_ind, lon_ind]
        v_US = v[:n_sample, lat_ind, lon_ind]

        # combine u, v, T into one input tensor of shape (num_samples, num_t, nlat, nlon, 3)
        X = np.zeros((n_sample, u_US.shape[1], u_US.shape[2], 3))
        print('input shape:', X.shape)

        X[:,:,:,0] = T_US
        X[:,:,:,1] = u_US
        X[:,:,:,2] = v_US

        #Scale data
        X = self.normalizer(X)

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

    #For denormalizing, use cur_scaler[j].inverse_transform. Cur_scaler is defined for each year, so use accordingly. 
    #Scaler Dict is defined for each type of input.
    def normalizer(self, X):
        self.X_mean = np.mean(X, axis=0)  # average over the time dimension
        self.X_std = np.std(X, axis=0)
        X = (X - self.X_mean)/self.X_std
#        self.X_scaled = np.zeros_like(self.X)
#        for i in range(3):
#            cur_scaler = {}
#            for j in range(self.X.shape[0]):
#                cur_scaler[j] = StandardScaler()
#                self.X_scaled[:,:,:,i] = cur_scaler[j].fit_transform(self.X[j,:,:,i])
#            self.scaler_dict[i] = cur_scaler
        return X

    def denormalizer(self, X):
        X = X*self.X_std + self.X_mean
        return X

    def model_setup(self):
        # TO DO: Make this more accessible so layers can be defined in an easier manner
        with tf.device(device):
          model = keras.Sequential(
            [
                keras.layers.Input(
                    shape=(self.num_time, self.rows, self.cols,self.channels), name="input_layer"
                ),
                keras.layers.ConvLSTM2D(
                    filters = 64, kernel_size = (2, 2), return_sequences = True, data_format = "channels_last", input_shape = (self.num_time, self.rows, self.cols,self.channels), padding="same", activation = self.activation
                ),               
                keras.layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=128, kernel_size=(2, 2), padding="same", return_sequences=True
                ),
                #layers.BatchNormalization(),
                #layers.ConvLSTM2D(
                #    filters=64, kernel_size=(2, 2), padding="same", return_sequences=True
                #),
                #layers.BatchNormalization(),
                #layers.ConvLSTM2D(
                #    filters=64, kernel_size=(3, 3), padding="same", return_sequences=True
                #),
                #layers.BatchNormalization(),
                layers.Conv3D(
                    filters=3, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )

          lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-2,
                        decay_steps=10000,
                        decay_rate=0.9)
          optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)
#           model.compile(loss = self.loss_fn,optimizer=optimizer, metrics = ['accuracy','mse'])
          model.train_step = self.make_print_data_and_train_step(model)
          model.compile(loss = self.loss_fn,optimizer=optimizer, metrics = ['accuracy','mse'],run_eagerly=True)

        self.model = model
        print(model.summary())
        return

    def make_print_data_and_train_step(self, keras_model):
        original_train_step = keras_model.train_step

        def print_data_and_train_step(original_data):
            # Basically copied one-to-one from https://git.io/JvDTv
            data = data_adapter.expand_1d(original_data)
            x, y_true, w = data_adapter.unpack_x_y_sample_weight(data)
            y_pred = keras_model(x, training=True)

            # this is pretty much like on_train_batch_begin
            # K.print_tensor(w, "Sample weight (w) =")
            # K.print_tensor(x, "Batch input (x) =")
            # K.print_tensor(y_true, "Batch output (y_true) =")
            # K.print_tensor(y_pred, "Prediction (y_pred) =")

            result = original_train_step(original_data)

            # add anything here for on_train_batch_end-like behavior
            self.targets.append(y_true.numpy())
            self.outputs.append(y_pred.numpy())
            # print(", Length of Targets",len(self.targets))
            # self.targets.append(tf.make_ndarray(K.eval(keras_model.input)))
            # self.outputs.append(tf.make_ndarray(K.eval(keras_model.input)))
            return result

        return print_data_and_train_step
     
    def training(self):
        self.history = self.model.fit(
            self.X_train_plus_val, self.Y_train_plus_val, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
            validation_split= 1- self.percent_train_val, shuffle = self.shuffle
        )
        history = self.history
        print(history.history.keys())
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        return
    

def main():
    convLstm = Model()
    convLstm.load_data()
    convLstm.model_setup()
    convLstm.training()

if __name__ == "__main__":
    main()

# For plotting, check if the data error was fixed!
# z = (m.cbk.targets[158] - m.cbk.outputs[158]).reshape(8*18,8,17,3)[143,:,:,0]
# x = m.lat_us
# y = m.lon_us
# import matplotlib.pyplot as plt
# plt.imshow(z, cmap='hot', interpolation='nearest')
# plt.colorbar()
