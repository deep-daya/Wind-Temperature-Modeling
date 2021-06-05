'''
This script is to 
1) train over the monthly U.S. data
2) plot loss (MSE) and metrics (MAPE) curves
3) plot relative difference map 
'''

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
from keras.callbacks import Callback
import math

USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'

class Model:
    def __init__(self, percent_train_test = 0.8, percent_train_val = 0.8, window_size = 12, activation = "relu",loss_fxn="MSE", 
    epochs = 200, batch_size = 24, verbose = 1, shuffle = False ):
        self.train_amount = percent_train_test
        self.percent_train_val = percent_train_val
        self.num_time = window_size
        self.activation = activation
        self.loss_fn = loss_fxn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.X_mean = {}
        self.X_std = {}
        #self.scaler_dict = {}
        #self.scaler = "Standard"


    def load_data(self):
        # read in data
        dir = './ncep_data/'
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

        X = X.reshape(int(X.shape[0]/self.num_time), self.num_time, X.shape[1],X.shape[2],X.shape[3])

        # partition train and test data
        #  80% train, 20% test
        num_train = int(X.shape[0]*self.train_amount)
        X_train_old = X[:num_train]
        Y_train_old = X[1:num_train + 1]
        X_test = self.normalizer_space(X[num_train:-1],'X_test')  # reserve the last time step as the label
        # shift train or test data by one time step to get the labels
        Y_test = self.normalizer_space(X[num_train+1:],'Y_test')
        # further divide train data into train and validation data
        num_train_sub = int(num_train*self.percent_train_val)
        
        # normalize data
        X_train = self.normalizer_space(X_train_old[:num_train_sub], "X_train")
        Y_train = self.normalizer_space(X_train_old[1 : num_train_sub+1], "Y_train")

        X_val = self.normalizer_space(X_train_old[num_train_sub : -1], "X_val")
        Y_val = self.normalizer_space(X_train_old[num_train_sub+1:], "Y_val")

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
    def normalizer_space(self, X,data_type):
        # self.X_mean = np.mean(X, axis=0)  # average over the time dimension
        # self.X_std = np.std(X, axis=0)
        # X = (X - self.X_mean)/self.X
        changed = False
        if len(X.shape)==5:
            Xn = X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
            changed = True
        print('in norm, xn shape', Xn.shape)
        self.X_mean[data_type] = np.mean(Xn, axis=(1,2))  # average over the time dimension
        self.X_std[data_type] = np.std(Xn, axis=(1,2))
        Xn = (Xn-self.X_mean[data_type][:,None,None,:])/self.X_std[data_type][:,None,None,:]
        if changed:
            Xn = Xn.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        # X = (X - self.X_mean)/self.X_std
#        self.X_scaled = np.zeros_like(self.X)
#        for i in range(3):
#            cur_scaler = {}
#            for j in range(self.X.shape[0]):
#                cur_scaler[j] = StandardScaler()
#                self.X_scaled[:,:,:,i] = cur_scaler[j].fit_transform(self.X[j,:,:,i])
#            self.scaler_dict[i] = cur_scaler
        return Xn

    def denormalizer_space(self, X, data_type):
        if len(X.shape)==5:
            X = X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])

        X = X*self.X_std[data_type][:,None,None,:] + self.X_mean[data_type][:,None,None,:]
        return X

    def normalizer_time(self, X, data_type):
        changed = False
        if len(X.shape)==5:
            Xn = X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])
            changed = True
        print('in norm x shape=', X.shape)
        self.X_mean[data_type] = np.mean(Xn, axis=0)  # average over the time dimension
        self.X_std[data_type] = np.std(Xn, axis=0)
        Xn = (Xn - self.X_mean[data_type])/self.X_std[data_type]
        
        if changed:
            Xn = Xn.reshape(X.shape[0],X.shape[1], X.shape[2], X.shape[3], X.shape[4])
        return Xn

    def denormalizer_time(self, X, data_type):
        if len(X.shape)==5:
            X = X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3], X.shape[4])

        X = X*self.X_std[data_type]+ self.X_mean[data_type]
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
                layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=64, kernel_size=(2, 2), padding="same", return_sequences=True
                ),
                layers.BatchNormalization(),
                layers.ConvLSTM2D(
                    filters=32, kernel_size=(3, 3), padding="same", return_sequences=True
                ),
                layers.BatchNormalization(),
                layers.Conv3D(
                    filters=3, kernel_size=(3, 3, 3), activation="relu", padding="same"
                ),
            ]
        )

          lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-2,
                        decay_steps=10000,
                        decay_rate=0.9)
          optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=0.5)
          model.compile(loss = self.loss_fn,optimizer=optimizer, metrics = ['mse'])

        self.model = model
        print(model.summary())
        return

    def training(self):
        #self.cbk = CollectOutputAndTarget()
        #fetches = [tf.assign(self.cbk.target, self.model._targets[0], validate_shape=False),
        #           tf.assign(self.cbk.output, self.model.outputs[0], validate_shape=False),
        #           tf.assign(self.cbk.input, self.model.inputs[0], validate_shape=False)]
        #self.model._function_kwargs = {'fetches': fetches}  
        self.history = self.model.fit(
            self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
           validation_data=(self.X_val, self.Y_val), shuffle = self.shuffle
        )
        history = self.history
        print(history.history.keys())
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'val'], loc='upper left')
        # plt.show()

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('plots/monthly_mape_12window_size_relu_norm_xy_4lay.png')
        return

    def plot_differences(self, y_true, y_pred, save_dir, data_type):
        # denormalize data
        y_true = self.denormalizer_space(y_true, data_type) #(time, lat, lon, 3)
        y_pred = self.denormalizer_space(y_pred, data_type)
        y_true = np.where(np.isnan(y_true), 0., y_true)
        y_pred = np.where(np.isnan(y_pred), 0., y_pred)

        try:
          z_T = (y_true - y_pred).reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2], y_true.shape[3],y_true.shape[4])[-1,:,:,0]
          z_U = (y_true - y_pred).reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2], y_true.shape[3],y_true.shape[4])[-1,:,:,1]
          z_V = (y_true - y_pred).reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2], y_true.shape[3],y_true.shape[4])[-1,:,:,2]
          # z_2 = (self.denormalizer(y_true, type_data = "Y_val")- self.denormalizer(y_pred, type_data = "Y_val")).reshape(y_true.shape[0]*y_true.shape[1], y_true.shape[2], y_true.shape[3],y_true.shape[4])
        except:
          print('in plot diff: y_true shape', y_true.shape)
          z_T = (y_true - y_pred)[-1,:,:,0]/np.max(np.abs(y_true[-1,:,:,0]))
          z_U = (y_true - y_pred)[-1,:,:,1]/np.max(np.abs(y_true[-1,:,:,1]))
          z_V = (y_true - y_pred)[-1,:,:,2]/np.max(np.abs(y_true[-1,:,:,2]))

          # z_2 = (self.denormalizer(y_true, type_data = "Y_val")- self.denormalizer(y_pred, type_data = "Y_val"))

        plt.figure()
        plt.imshow(z_T, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label(r'$(T_{true}-T_{pred})/max(|T_{true}|)$')
        plt.savefig(save_dir + "/monthly_diff_T_12winsize_norm_xy_relu_4lay.png")

        plt.figure()
        plt.imshow(z_U, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label(r'$(u_{true}-u_{pred})/max(|u_{true}|$')
        plt.savefig(save_dir + "/monthly_diff_U_12winsize_norm_xy_relu_4lay.png")

        plt.figure()
        plt.imshow(z_V, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label(r'$(v_{true}-v_{pred})/max(|v_{true}|$')
        plt.savefig(save_dir + "/monthly_diff_V_12winsize_norm_xy_relu_4lay.png")

    def predict(self, test_time, x_test, y_test, us2, data_type, model_name, model=None):
        # load model
        if test_time:
            model = keras.models.load_model(model_name) 
        else:
            try:
                model = self.model
            except:
                model = keras.models.load_model(model_name) 

        #x_test = self.normalizer_time(x_test, 'X'+data_type)
        #y_test = self.normalizer_time(y_test, 'Y'+data_type)

        y_pred = model.predict(x_test)
        y_pred = np.where(y_pred<1e-6, np.nan, y_pred)

        if us2:
            y_test = y_test[:,:, 2:-2, 2:-2, :]
            y_pred2 = y_pred[:,:, 2:-2, 2:-2, :]
        else:
            y_pred2 = y_pred
        mape =  np.nanmean(np.abs(y_pred2 - y_test)/np.abs(y_test))*100
        #print('y test T diff', np.abs(y_pred[-1,:,:,0]- y_test[-1,:,:,0]))
        #print('y test u', np.abs(y_pred[-1,:,:,1]-y_test[-1,:,:,1]))
        #print('y test v', np.abs(y_pred[-1,:,:,2]-y_test[-1,:,:,2]))

        err_mse = np.nanmean((y_pred2 - y_test)**2)
        if test_time:
            print('Test predicted mse', err_mse)
            print('Test predicted MAPE:', mape)
        else:
            print('Validation mse', err_mse)
            print('Validation MAPE:', mape)

        return (y_pred, err_mse, mape) 
    
    
#class CollectOutputAndTarget(Callback):
#    def __init__(self):
#        super(CollectOutputAndTarget, self).__init__()
#        self.targets = []  # collect y_true batches
#        self.outputs = []  # collect y_pred batches
#        self.inputs = []
#
#        # the shape of these 2 variables will change according to batch shape
#        # to handle the "last batch", specify `validate_shape=False`
#        self.input = tf.Variable(0.0, shape=tf.TensorShape(None))
#        self.target = tf.Variable(0.0, shape=tf.TensorShape(None))
#        self.output = tf.Variable(0.0, shape=tf.TensorShape(None))
#
#    def on_batch_end(self, batch, logs=None):
#        # evaluate the variables and save them into lists
#        self.inputs.append(K.eval(self.input))
#        self.targets.append(K.eval(self.target))
#        self.outputs.append(K.eval(self.output))
       
def main():
    train = True
    test = True
    us2 = False
    convLstm = Model()
    data = convLstm.load_data()
    if train:
        save_dir = 'plots'
        convLstm.model_setup()
        convLstm.training()
        model_name = 'models/monthly_norm_xy_12winsize_relu_4lay'
        (convLstm.model).save(model_name)
        convLstm.predict(True, convLstm.X_test, convLstm.Y_test,us2,'_test', model_name) # test data prediction
        y_pred, _ ,_ = convLstm.predict(False, convLstm.X_val, convLstm.Y_val, us2, '_val', model_name) # val data prediction

        # plot validation prediction vs trueth
        convLstm.plot_differences(convLstm.Y_val, y_pred, save_dir,'Y_val')
    
    if test:
        save_dir = 'plots'
        y_pred, _ ,_ = convLstm.predict(False, convLstm.X_val, convLstm.Y_val, us2, '_val', model_name) # val data prediction

        # plot validation prediction vs trueth
        convLstm.plot_differences(convLstm.Y_val, y_pred, save_dir,'Y_val')

        save_dir = 'test_month_plots'
        y_pred,_,_=convLstm.predict(True, convLstm.X_test, convLstm.Y_test, us2,'_test', model_name) # test data prediction
        convLstm.plot_differences(convLstm.Y_test, y_pred, save_dir,'Y_test')


if __name__ == "__main__":
    main()

# For plotting, check if the data error was fixed!
# z = (m.cbk.targets[158] - m.cbk.outputs[158]).reshape(8*18,8,17,3)[143,:,:,0]
# x = m.lat_us
# y = m.lon_us
# import matplotlib.pyplot as plt
# plt.imshow(z, cmap='hot', interpolation='nearest')
# plt.colorbar()
