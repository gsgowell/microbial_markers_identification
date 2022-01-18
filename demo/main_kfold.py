import scipy.io as sio
import numpy as np
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Activation, initializers, Dropout, regularizers
from keras.models import Sequential, save_model, load_model
from sklearn import preprocessing
from keras.callbacks import History, EarlyStopping
from keras.utils import to_categorical
import os
import math
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score

# data.mat; 
datafile = './d_368.mat'
dataset = sio.loadmat(datafile)
X = dataset['X_368_rf'][:, 0:40] # X_368, X_368_LEfSe,['X_368_rf']
Y = dataset['Y_368'].ravel()  # 



nn = 1
num_classes = 2
(nsize, nf) = X.shape

X = np.array(X)
Y = np.array(Y)
# Y = to_categorical(Y,num_classes)
acc = []
f1 = []
recall = []
precision = []
#  ####################################################          set model
batch_size = 3  # 2, 4, 5, 6
epochs = 15   # 15, 20, 10, 25, 30


def construct_ann_model():
    model = Sequential()
    model.add(Dense(input_dim=nf, units=16, kernel_initializer=initializers.glorot_normal()
                    , kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(8, kernel_initializer=initializers.glorot_normal()))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))  # tanh,relu
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    return model


##########################################################

j = 0
mean_tpr_svm_r = 0
mean_fpr_svm_r = np.linspace(0, 1, 100)
aucs_nn = []

ann_model = construct_ann_model()
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_linear_model = svm.SVC(kernel='linear', probability=True, gamma='auto')
rf_model = RandomForestClassifier(n_estimators=500)
svm_rbf_model = svm.SVC(kernel='rbf', probability=True, gamma='auto')

kf = KFold(n_splits=5, shuffle=True, random_state=2)# shuffle=False , random_state=2, 3, 4
for train_ind, test_ind in kf.split(X, Y):
    X_test, Y_test = X[test_ind], Y[test_ind]
    X_train, Y_train = X[train_ind], Y[train_ind]

    scaler = preprocessing.MinMaxScaler()  # MinMaxScaler(), StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n = scaler.transform(X_test)


    Y_test_n = to_categorical(Y_test, num_classes)
    Y_train_n = to_categorical(Y_train, num_classes)

    ann_model.fit(X_train_n, Y_train_n, batch_size=batch_size, epochs=epochs, verbose=0)
    # knn_model.fit(X_train_n, Y_train)
    # rf_model.fit(X_train_n, Y_train)
    # svm_linear_model.fit(X_train_n, Y_train)
    # svm_rbf_model.fit(X_train_n, Y_train)

    ''''''
    p_n = ann_model.predict_proba(X_test_n)
    # p_n = knn_model.predict_proba(X_test_n)
    # p_n = rf_model.predict_proba(X_test_n)
    # p_n = svm_linear_model.predict_proba(X_test_n)
    # p_n = svm_rbf_model.predict_proba(X_test_n)

    fpr_n, tpr_n, thresholds_n = roc_curve(Y_test, p_n[:, 1])
    mean_tpr_svm_r += interp(mean_fpr_svm_r, fpr_n, tpr_n)
    mean_tpr_svm_r[0] = 0.0
    roc_auc = auc(fpr_n, tpr_n)
    aucs_nn.append(roc_auc)

    y_pred = ann_model.predict_classes(X_test_n)
    # y_pred = knn_model.predict(X_test_n)
    # y_pred = rf_model.predict(X_test_n)
    # y_pred = svm_linear_model.predict(X_test_n)
    # y_pred = svm_rbf_model.predict(X_test_n)

    acc.append(accuracy_score(Y_test, y_pred))
    precision.append(precision_score(Y_test, y_pred))
    f1.append(f1_score(Y_test, y_pred))
    recall.append(recall_score(Y_test, y_pred))

print(acc)
print('acc_mean:', np.mean(acc))
print(precision)
print('precision_mean:', np.mean(precision))
print(f1)
print('f1_mean:', np.mean(f1))
print(recall)
print('recall_mean:', np.mean(recall))
print(aucs_nn)
print('roc_auc:', np.mean(aucs_nn))
print('auc_std:', np.std(aucs_nn))
'''
mean_tpr_svm_r /= 5 * nn
mean_tpr_svm_r[-1] = 1.0
mean_auc_ann = auc(mean_fpr_svm_r, mean_tpr_svm_r)
std_nn = np.std(aucs_nn)

np.save("mean_tpr_ann", mean_tpr_svm_r)
np.save("mean_fpr_ann", mean_fpr_svm_r)
np.save("mean_auc_ann", mean_auc_ann)
np.save("std_ann", std_nn)'''

'''
plt.plot(mean_tpr_svm_r,mean_fpr_svm_r, label = "KNN (Mean_AUC = %0.2f $\pm$ %0.2f)" % (mean_auc_nn, std_nn)) # $\pm$ %0.2f
plt.title('ROC curves')
plt.xlim([0.0,1.0])
plt.ylim([0,1.05])
plt.xlabel('Fase Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
'''


