import scipy.io as sio
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation
from keras.models import Sequential, save_model, load_model
from sklearn import preprocessing
from keras.callbacks import History, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

datafile = './d_68.mat'
dataset = sio.loadmat(datafile)

X = dataset['X_368_rf'][:, 0: 40]
Y = dataset['Y_fbg']

X = np.array(X)
Y = np.array(Y)
(nsize, nf) = X.shape
# normalize the train dataset
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = 2*X_std - 1
Y_std = (Y - Y.min(axis=0)) / (Y.max(axis=0) - Y.min(axis=0))

num_eochs = 25
batch_s = 2

model = Sequential()
model.add(Dense(input_dim=nf, units=16))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(loss='mse',
              optimizer=Adam())  # categorical_crossentropy  mse binary_crossentropy

yp = []
yr = []

kf = KFold(n_splits=5)
for train_ind, test_ind in kf.split(X_std, Y_std):
    X_test, Y_test = X_std[test_ind], Y_std[test_ind]
    X_train, Y_train = X_std[train_ind], Y_std[train_ind]
    model.fit(X_train, Y_train, batch_size=batch_s, epochs=num_eochs, verbose=1)
    y_t = model.predict(X_test)
    yp.extend(y_t)
    yr.extend(Y_test)
# y_p = 0.5*(y_tst + 1)*(Y.max(axis=0)-Y.min(axis=0)) + Y.min(axis=0);

y_p = yp * (Y.max(axis=0) - Y.min(axis=0)) + Y.min(axis=0)
y_r = yr * (Y.max(axis=0) - Y.min(axis=0)) + Y.min(axis=0)
# y_p = y_tst
# R = np.corrcoef(y_p,tstY_i2)
# print(y_p)
r1 = np.array(y_p).flatten()
r2 = np.array(y_r).flatten()

R = np.corrcoef(r1, r2)

print(R[0][1])

W = model.get_weights()
sio.savemat('diabet_368_fbg_regression_rf40_kfold.mat', {'W': W, 'R': R[0][1], 'y_ps': y_p, 'y_rs': Y})
