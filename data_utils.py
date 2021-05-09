import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

def load_data():
    # read in data
    data = Dataset('../air.mon.mean.nc', 'r')
    T = data.variables['air'][:,0]
    t = data.variables['time'][:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()

    data = Dataset('../uwnd.mon.mean.nc', 'r')
    u = data.variables['uwnd'][:,0]
    data.close()

    data = Dataset('../vwnd.mon.mean.nc', 'r')
    v = data.variables['vwnd'][:,0]
    data.close()
    
    T = np.flip(T, axis=1)
    u = np.flip(u, axis=1)
    v = np.flip(v, axis=1)

    # extract data of the US
    lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
    lon_us = np.argwhere(np.logical_and(lon > np.mod(-123.3,360),lon < np.mod(-81.2, 360)))[:,0]
    lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

    T_US = T[:, lat_ind, lon_ind]
    u_US = u[:, lat_ind, lon_ind]
    v_US = v[:, lat_ind, lon_ind]

    # combine u, v, T into one input tensor of shape (num_t, nlat, nlon, 3)
    X = np.zeros((u_US.shape[0], u_US.shape[1], u_US.shape[2], 3))
    print('input shape:', X.shape)

    X[:,:,:,0] = T_US
    X[:,:,:,1] = u_US
    X[:,:,:,2] = v_US


    # partition train and test data
      #  80% train, 20% test
    num_train = int(len(t)*0.8)
    X_train_old = X[:num_train]
    X_test = X[num_train:-1]  # reserve the last time step as the label
      # shift train or test data by one time step to get the labels
    Y_test = X[num_train+1:]

    # further divide train data into train and validation data
    num_train_sub = int(num_train*0.8)
    X_train = X_train_old[:num_train_sub]
    Y_train = X_train_old[1 : num_train_sub+1]
    X_val = X_train_old[num_train_sub : -1]
    Y_val = X_train_old[num_train_sub+1:]

    print('x train shape:', X_train.shape)
    print('y train shape:', Y_train.shape)
    print('x val shape:', X_val.shape)
    print('y val shape:', Y_val.shape)
    print('x test shape:', X_test.shape)
    print('y test shape:', Y_test.shape)


    return {"X_train": X_train, "Y_train": Y_train, \
            "X_val":X_val, "Y_val": Y_val,\
            "X_test":X_test, "Y_test":Y_test}

