def load_data(self):
    # read in data
    data = Dataset('./data_modeling/assembled_file_final.nc', 'r')
    T = data['T'][:,0]
    u = data['U'][:,0]
    v = data['V'][:,0]
    t = data.variables['time'][:]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    data.close()

    T_US = np.flip(T, axis=1)
    u_US = np.flip(u, axis=1)
    v_US = np.flip(v, axis=1)

    # extract data of the US
    # lat_us = np.argwhere(np.logical_and(lat > 29.06, lat<48.97))[:,0]
    # lon_us = np.argwhere(np.logical_and(lon > np.mod(-123.3,360),lon < np.mod(-81.2, 360)))[:,0]
    # lon_ind, lat_ind = np.meshgrid(lon_us, lat_us)

    # T_US = T[:-4, lat_ind, lon_ind]
    # u_US = u[:-4, lat_ind, lon_ind]
    # v_US = v[:-4, lat_ind, lon_ind]

    # combine u, v, T into one input tensor of shape (num_samples, num_t, nlat, nlon, 3)
    X = np.zeros((u_US.shape[0], u_US.shape[1], u_US.shape[2], 3))
    print('input shape:', X.shape)

    X[:,:,:,0] = T_US
    X[:,:,:,1] = u_US
    X[:,:,:,2] = v_US
    X = X[:-3,:,:,:]
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
