'''
This script is to 
1) train over the daily U.S.+surrounding dataset
2) plot loss (MSE) and metrics (MAPE) curves
3) plot relative difference map 
4) time series on the validation data
'''



import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
import keras.backend as K
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.engine import data_adapter
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
    def __init__(self, percent_train_test = 0.8, percent_train_val = 0.8, window_size = 7, activation = "relu",loss_fxn="MSE", drop_rate = 0.5, epochs = 50, batch_size = 128, verbose = 1, shuffle = False ):
        self.train_amount = percent_train_test
        self.percent_train_val = percent_train_val
        self.num_time = window_size
        self.activation = activation
        self.loss_fn = loss_fxn
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.X_mean = {}
        self.X_std = {}
        self.batch_num = 0
        self.mape = []
        self.mape_T = []
        self.mape_u = []
        self.mape_v = []
        self.val_mape = []
        self.mape_val = []
        self.mape_T_val = []
        self.mape_u_val = []
        self.mape_v_val = []
        #self.scaler_dict = {}
        #self.scaler = "Standard"


    def load_data(self):
        # read in data
        dir = './ncep_data/'
        data = Dataset(dir + 'US_daily.nc', 'r')
        #data = Dataset("/content/drive/MyDrive/CS231N-Project/US_daily.nc")
        T = data.variables['T2'][:]
        t = data.variables['TIME'][:]
        u = data.variables['u2'][:]
        v = data.variables['v2'][:]
        lat = data.variables['lat2'][:]
        lon = data.variables['lon2'][:]
        data.close()

        n_seq = int(T.shape[0]/self.num_time) 
        N = n_seq*self.num_time
        print('******num samp=', N,'*******')

        # combine u, v, T into one input tensor of shape (num_samples, num_t, nlat, nlon, 3)
        X = np.zeros((N, u.shape[1], u.shape[2], 3))
        print('input shape:', X.shape)

        X[:,:,:,0] = T[:N]
        X[:,:,:,1] = u[:N]
        X[:,:,:,2] = v[:N]

        #Scale data
        # X = self.normalizer_space(X)

        X = X.reshape(n_seq, self.num_time, X.shape[1],X.shape[2],X.shape[3])

        # partition train and test data
        #  80% train, 20% test
        num_train = int(X.shape[0]*self.train_amount)
        X_train_old = X[:num_train]
        Y_train_old = X[1:num_train + 1]
        X_test = self.normalizer_space(X[num_train:-1], type_data = "X_test")  # reserve the last time step as the label
        # shift train or test data by one time step to get the labels
        Y_test = self.normalizer_space(X[num_train+1:], type_data = "Y_test")
        # further divide train data into train and validation data
        num_train_sub = int(num_train*self.percent_train_val)
        
        X_train = self.normalizer_space(X_train_old[:num_train_sub], type_data = "X_train")
        Y_train = self.normalizer_space(X_train_old[1 : num_train_sub+1], type_data = "Y_train")
        
        X_val = self.normalizer_space(X_train_old[num_train_sub : -1], type_data = "X_val")
        Y_val = self.normalizer_space(X_train_old[num_train_sub+1:], type_data = "Y_val")
        self.X_train_plus_val = X_train_old
        self.Y_train_plus_val = Y_train_old
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_test = X_test  # 5 dims
        self.Y_test = Y_test
        self.channels = X.shape[4]
        self.rows  = X.shape[2]
        self.cols = X.shape[3]
        self.time = t
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
    def normalizer_year(self, X):
        # average for each year
        nt, nlat, nlon, nvar = X.shape
        nyear = nt//365
        print('nt=', nt, 'nyear=', nyear)
        n_toavg = 365*nyear
        X_temp = X[:n_toavg].reshape(365, nyear, nlat, nlon, nvar)
        self.X_mean = np.mean(X_temp, axis=0)  # average over the time dimension
        self.X_std = np.std(X_temp, axis=0)
        print('x mean shape', self.X_mean.shape)
        X_temp = (X_temp - self.X_mean)/self.X_std
        X_temp = X_temp.reshape(n_toavg, nlat, nlon, nvar)
        X[:n_toavg] = X_temp

        if nt%365:
            X_left = X[n_toavg:]
            left_mean = np.mean(X_left, axis=0)
            left_std = np.std(X_left, axis=0)
            print('left mean shape', left_mean[None,:].shape)
            self.X_mean = np.append(self.X_mean, left_mean[None,:], axis=0)
            self.X_std = np.append(self.X_std, left_std[None,:], axis=0)
            X[n_toavg:] = (X_left-left_mean)/left_std

#        self.X_scaled = np.zeros_like(self.X)
#        for i in range(3):
#            cur_scaler = {}
#            for j in range(self.X.shape[0]):
#                cur_scaler[j] = StandardScaler()
#                self.X_scaled[:,:,:,i] = cur_scaler[j].fit_transform(self.X[j,:,:,i])
#            self.scaler_dict[i] = cur_scaler
        return X

    def normalizer_space(self, X, type_data="X_train"):
        changed =False
        if len(X.shape)== 5:
          Xn = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4])
          changed = True
        else: 
          Xn = X
        self.X_mean[type_data] = np.mean(Xn, axis=(1, 2))
        self.X_std[type_data] = np.std(Xn, axis=(1, 2))
        Xn = (Xn- self.X_mean[type_data][:, None, None, :])/self.X_std[type_data][:, None, None, :]
        if changed:
          X = Xn.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],X.shape[4])
        return X

    def denormalizer_space(self, X, type_data = "X_train"):
        '''
        input X: either 4 or 5 dimensional
        output Xn: 4 dimensional (time, lat, lon, 3)
        '''
        changed =False
        if len(X.shape)== 5:
           Xn = X.reshape(X.shape[0]*X.shape[1],X.shape[2],X.shape[3],X.shape[4])
           print(Xn.shape)
           changed = True
        else:
           Xn = X
        Xn = Xn * self.X_std[type_data][:, None, None, :] + self.X_mean[type_data][:, None, None, :]
        #if changed:
        #   X = Xn.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],X.shape[4])
        return Xn

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
                #keras.layers.BatchNormalization(),
                keras.layers.Dropout(self.drop_rate),         
                layers.ConvLSTM2D(
                    filters=128, kernel_size=(2, 2), padding="same", return_sequences=True
                ),
                #keras.layers.BatchNormalization(),
                keras.layers.Dropout(self.drop_rate),
                # layers.ConvLSTM2D(
                #     filters=64, kernel_size=(2, 2), padding="same", return_sequences=True
                # ),
                # layers.BatchNormalization(),
                # layers.ConvLSTM2D(
                #     filters=64, kernel_size=(3, 3), padding="same", return_sequences=True
                # ),
                # layers.BatchNormalization(),
                 layers.Conv3D(
                     filters=3, kernel_size=(3, 3, 3), activation="relu", padding="same"
                 ),
            ]
        )

          lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=5e-3,
                        decay_steps=10000,
                        decay_rate=0.9)
          optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)# clipvalue=0.5)
          model.train_step = self.make_print_data_and_train_step(model)
          model.compile(loss = self.loss_fn,optimizer=optimizer, metrics = ['mean_absolute_percentage_error','mse'],run_eagerly=True)

        self.model = model
        print(model.summary())
        return

    def make_print_data_and_train_step(self, keras_model):
        original_train_step = keras_model.train_step

        def mape(y_pred, y_test):
            y_test = np.where(y_test<1e-6, np.nan, y_test)
            return np.nanmean(np.abs(y_pred - y_test)/np.abs(y_test))*100

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
            if self.batch_num % math.ceil(self.X_train.shape[0]/self.batch_size) == 0: 
              # only compute data within the US
              y_true = y_true[:,:, 2:-2, 2:-2, :]
              y_pred = y_pred[:,:, 2:-2, 2:-2, :]

              err_mse = np.mean((y_pred - y_true)**2)
              
              mape_all = mape(y_pred, y_true)
              mape_T = mape(y_pred[:,:,:,:,0],y_true[:,:,:,:,0])
              mape_U = mape(y_pred[:,:,:,:,1],y_true[:,:,:,:,1])
              mape_V = mape(y_pred[:,:,:,:,2],y_true[:,:,:,:,2])
              y_pred_val = keras_model(self.X_val, training = False) 
              err_mse_Val = np.mean((y_pred_val - self.Y_val)**2)
              mape_all_val = mape(y_pred_val, self.Y_val)
              mape_T_val = mape(y_pred_val[:,:,:,0],self.Y_val[:,:,:,0])
              mape_U_val = mape(y_pred_val[:,:,:,1],self.Y_val[:,:,:,1])
              mape_V_val = mape(y_pred_val[:,:,:,2],self.Y_val[:,:,:,2])
              # print(y_pred.shape)
              # y_pred_denorm = self.denormalizer(tf.convert_to_tensor(y_pred.numpy()), type_data = "Y_val")
              # y_true_denorm = self.denormalizer(y_true, type_data = "Y_val")
              # mape_all_denorm = mape(y_pred_denorm, y_true_denorm)
              # mape_T_denorm = mape(y_pred_denorm[:,:,:,0],y_true_denorm[:,:,:,0])
              # mape_U_denorm = mape(y_pred_denorm[:,:,:,1],y_true_denorm[:,:,:,1])
              # mape_V_denorm = mape(y_pred_denorm[:,:,:,2],y_true_denorm[:,:,:,2])
              print('Predicted mse', err_mse)
              print('MAPE ALL Normalized:', mape_all)
              print('MAPE T Normalized:', mape_T)
              print('MAPE U Normalized:', mape_U)
              print('MAPE V Normalized:', mape_V)
              print("Predicted MSE Val",err_mse_Val)
              print('MAPE ALL Normalized VAL:', mape_all_val)
              print('MAPE T Normalized  VAL:', mape_T_val)
              print('MAPE U Normalized VAL:', mape_U_val)
              print('MAPE V Normalized VAL:', mape_V_val)
              self.mape.append(mape_all)
              self.mape_T.append(mape_T)
              self.mape_u.append(mape_U)
              self.mape_v.append(mape_V)
              self.mape_val.append(mape_all_val)
              self.mape_T_val.append(mape_T_val)
              self.mape_u_val.append(mape_U_val)
              self.mape_v_val.append(mape_V_val)
              # compute validation data metrics
              #test_time = False # it is validation time
              #_, val_mape = self.predict(test_time, self.X_val, self.Y_val, keras_model)
              #self.val_mape.append(val_mape)


              # print('MAPE ALL Denormalized:', mape_all_denorm)
              # print('MAPE T Denormalized:', mape_T_denorm)
              # print('MAPE U Denormalized:', mape_U_denorm)
              # print('MAPE V Denormalized:', mape_V_denorm)
            # print(", Length of Targets",len(self.targets))
            # self.targets.append(tf.make_ndarray(K.eval(keras_model.input)))
            # self.outputs.append(tf.make_ndarray(K.eval(keras_model.input)))
            self.batch_num += 1
            return result

        return print_data_and_train_step

    def training(self):
        self.history = self.model.fit(
            self.X_train, self.Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
            validation_data=(self.X_val, self.Y_val), shuffle = self.shuffle
        )

        history = self.history
        print(history.history.keys())

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('plots_us2/mse_us2_7window_size_3layers_nobatch.png')

        plt.figure()
        plt.plot(history.history['mean_absolute_percentage_error'])
        plt.plot(history.history['val_mean_absolute_percentage_error'])
        plt.ylabel('MAPE')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('plots_us2/mape_us2_7window_size_3layers_nobatch.png')

        plt.figure()
        plt.plot(self.mape, label='train')
        plt.plot(self.mape_val, label='val')
        plt.ylabel('MAPE')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig('plots_us2/mape2_us2_7window_size_3layers_nobatch.png')

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
        plt.savefig(save_dir + "/diff_us2_T_nobatch_7winsize.png")

        plt.figure()
        plt.imshow(z_U, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label(r'$(u_{true}-u_{pred})/max(|u_{true}|)$')
        plt.savefig(save_dir + "/diff_us2_U_nobatch_7winsize.png")

        plt.figure()
        plt.imshow(z_V, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label(r'$(v_{true}-v_{pred})/max(|v_{true}|)$')
        plt.savefig(save_dir + "/diff_us2_V_nobatch_7winsize.png")

    def predict(self, test_time, x_test, y_test, us2, data_type, model=None):
        # load model
        if test_time:
            model = keras.models.load_model("models/us2_norm_xy_7winsize_3layers_nobatch") 
        else:
            try:
                model = self.model
            except:
                model = keras.models.load_model("models/us2_norm_xy_7winsize_3layers_nobatch") 
        '''
         x_test and y_test are already normalized!!
        '''
        #x_test = self.normalizer_space(x_test, 'X'+data_type)
        #y_test = self.normalizer_space(y_test, 'Y'+data_type)

        y_pred = model.predict(x_test)
        y_pred = np.where(y_pred<1e-6, np.nan, y_pred)

        if us2:
            y_test = y_test[:,:, 2:-2, 2:-2, :]
            y_pred2 = y_pred[:,:, 2:-2, 2:-2, :]
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

    def plot_time_series(self, y_true, y_pred, us2, data_type, save_dir):
        y_true = self.denormalizer_space(y_true, data_type) #(time, lat, lon, 3)
        y_pred = self.denormalizer_space(y_pred, data_type)
        y_true = np.where(np.isnan(y_true), 0., y_true)
        y_pred = np.where(np.isnan(y_pred), 0., y_pred)

        # get the time axis
        if data_type == 'Y_val':
            t_end = -((self.Y_test).shape[0]*self.num_time + 1)
            t_start = t_end - y_true.shape[0]
            t = np.round( (self.time[t_start:t_end] - self.time[0])/(24*365), 2)+1948
        if data_type == 'Y_test':
            t_start = -y_true.shape[0]
            t = np.round( (self.time[t_start:] - self.time[0])/(24*365), 2)+1948
        print('t=', t.shape)
        # plot the average of the U.S.
        if us2:
            y_true = y_true[:, 2:-2, 2:-2, :]
            y_pred = y_pred[:, 2:-2, 2:-2, :]

        plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(t, np.mean(y_true[:,:,:,0], axis=(1, 2)), label='Truth',linewidth=0.5)
        plt.plot(t, np.mean(y_pred[:,:,:,0], axis=(1, 2)), label='Prediction',linewidth=0.5)
        plt.legend()
        plt.ylabel('T', fontsize=16)

        plt.subplot(312)
        plt.plot(t, np.mean(y_true[:,:,:,1], axis=(1, 2)), label='Truth',linewidth=0.5)
        plt.plot(t, np.mean(y_pred[:,:,:,1], axis=(1, 2)), label='Prediction',linewidth=0.5)
        plt.legend()
        plt.ylabel('u', fontsize=16)

        plt.subplot(313)
        plt.plot(t, np.mean(y_true[:,:,:,2], axis=(1, 2)), label='Truth',linewidth=0.5)
        plt.plot(t, np.mean(y_pred[:,:,:,2], axis=(1, 2)), label='Prediction',linewidth=0.5)
        plt.ylabel('v', fontsize=16)
        plt.xlabel('year', fontsize=18)
        plt.suptitle('U.S. average', fontsize=16)
        plt.legend()
        plt.savefig(save_dir +'/tseries_havg.png')

        # plot t series at a point
        plt.figure(figsize=(20,10))
        plt.subplot(311)
        lat = 4
        lon= 8
        plt.plot(t, y_true[:,lat, lon,0], label='Truth')
        plt.plot(t, y_pred[:, lat, lon,0], label='Prediction',linewidth=0.5)
        plt.ylabel('T', fontsize=16)
        plt.legend()

        plt.subplot(312)
        plt.plot(t, y_true[:,lat, lon,1], label='Truth')
        plt.plot(t, y_pred[:,lat, lon,1], label='Prediction',linewidth=0.5)
        plt.ylabel('u', fontsize=16)
        plt.legend()

        plt.subplot(313)
        plt.plot(t, y_true[:,lat, lon, 2], label='Truth',linewidth=0.5)
        plt.plot(t, y_pred[:,lat, lon,2], label='Prediction', linewidth=0.5)
        plt.ylabel('v', fontsize=16)
        plt.xlabel('t', fontsize=16)
        plt.legend()
        plt.suptitle('U.S. average', fontsize=16)
        plt.savefig(save_dir +'/tseries_point.png')

def main():
    train = False
    test = True
    us2 = True
    convLstm = Model()
    data = convLstm.load_data()
    if train:
        save_dir = 'val_plots_us2'
        convLstm.model_setup()
        convLstm.training()
        (convLstm.model).save('models/us2_norm_xy_7winsize_3layers_nobatch')
        convLstm.predict(True, convLstm.X_test, convLstm.Y_test,us2, '_test') # test data prediction
        y_pred, _ ,_ = convLstm.predict(False, convLstm.X_val, convLstm.Y_val, us2, '_val') # val data prediction

        # plot validation prediction vs trueth
        convLstm.plot_differences(convLstm.Y_val, y_pred, save_dir,'Y_val')
    if test:
        save_dir = 'val_plots_us2'
        y_pred, _ ,_ = convLstm.predict(False, convLstm.X_val, convLstm.Y_val, us2, '_val') # val data prediction
        # plot validation prediction vs trueth
        convLstm.plot_differences(convLstm.Y_val, y_pred, save_dir,'Y_val')
        convLstm.plot_time_series(convLstm.Y_val, y_pred, us2, 'Y_val', save_dir)

        save_dir = 'test_plots_us2'
        y_pred,_,_=convLstm.predict(True, convLstm.X_test, convLstm.Y_test, us2, '_test') # test data prediction
        convLstm.plot_differences(convLstm.Y_test, y_pred, save_dir,'Y_test')
        convLstm.plot_time_series(convLstm.Y_test, y_pred, us2, 'Y_test', save_dir)

if __name__ == "__main__":
    main()

# For plotting, check if the data error was fixed!
# z = (m.cbk.targets[158] - m.cbk.outputs[158]).reshape(8*18,8,17,3)[143,:,:,0]
# x = m.lat_us
# y = m.lon_us
# import matplotlib.pyplot as plt
# plt.imshow(z, cmap='hot', interpolation='nearest')
# plt.colorbar()
