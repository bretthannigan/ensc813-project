from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('./keras-gcn/'))
sys.path.append(os.path.abspath('./keras-gcn/tests'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout, Concatenate, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split

from keras_gcn import GraphConv, GraphMaxPool, GraphAveragePool

import pickle as pkl

import time
import argparse

from ReactionData import DataGenerator

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="KagglelogP.pickle",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=50,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=256,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.1,
                help="Learning rate")
ap.add_argument("-l2", "--l2norm", type=float, default=0.,
                help="L2 normalization of input weights")
ap.add_argument("-tf", "--trainfrac", type=float, default=0.8,
                help="Fraction of dataset used for training")

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('--validation', dest='validation', action='store_true')
fp.add_argument('--testing', dest='validation', action='store_false')
ap.set_defaults(validation=True)

args = vars(ap.parse_args())
print(args)

# Define parameters
DATASET = args['dataset']
NB_EPOCH = args['epochs']
VALIDATION = args['validation']
LR = args['learnrate']
L2 = args['l2norm']
HIDDEN = args['hidden']
BASES = args['bases']
DO = args['dropout']
TRAIN_FRACTION = args['trainfrac']

# with open(DATASET, 'rb') as f:
#     reactant = pkl.load(f)
#     %product = pkl.load(f)

# A = [r.get_adjacency(normalize=True) for r in reactant]
# X = [r.get_features() for r in reactant]
# y = [p.get_adjacency(normalize=True) for p in product]
# X_y = [p.get_features() for p in product]

with open(DATASET, 'rb') as f:
    compound = pkl.load(f)
    log_p = pkl.load(f)

A = [c.get_adjacency(normalize=True) for c in compound]
X = [c.get_features() for c in compound]
y = np.asarray(log_p)

A_train, A_test, X_train, X_test, y_train, y_test = train_test_split(A, X, y, train_size=TRAIN_FRACTION, random_state=0, shuffle=True)

num_nodes = A[0].shape[0]
support = 1

# training_data_generator = DataGenerator(A_train, X_train, y_train, X_y_train, batch_size=128)
# validation_data_generator = DataGenerator(A_test, X_test, y_test, X_y_test, batch_size=128)

# Define model architecture

X_in = Input(shape=(None, X_train[0].shape[1]), name='Input-Features')
A_in = Input(shape=(None, A_train[0].shape[1]), sparse=False, name='Input-Adjacency')
H = GraphConv(units=1024, step_num=1, name='GraphConv-1',activation='relu')([X_in, A_in])
#H = Dropout(DO)(H)
H = GraphConv(units=1024, step_num=1, name='GraphConv-2',activation='relu')([H, A_in])
#H = Dropout(DO)(H)
#H = Concatenate(axis=2)([X_in, A_in])
H = Dense(units=512, activation='tanh', name='Dense-1')(H)
H = Dense(units=256, activation='tanh', name='Dense-2')(H)
H = Dense(units=1, activation='linear', name='Dense-3')(H)
# Y = Dense(units=146, activation='linear')(H)
# H = GraphConv(units=1, step_num=1, name='GraphConv-3',activation='relu')([H, A_in])
Y = GlobalAveragePooling1D(data_format='channels_last', name='Global-Pooling')(H)
#Y = GraphConv(units=146, step_num=1, name='GraphConv-3',activation='linear')([H, A_in])

# Compile model
model = Model(inputs=[X_in, A_in], outputs=Y)
model.compile(loss='mse', optimizer=Adam(lr=LR))
model.summary()

# tr = [np.expand_dims(X_train[1].todense(), 0), np.expand_dims(A_train[1].todense(), 0)]
# yd = np.expand_dims(np.concatenate((X_y_train[1].todense(), y_train[1].todense()), axis=1), 0)

x_t = [np.asarray(x.todense()) for x in X_train]
a = [np.asarray(a.todense()) for a in A_train]
x_t_val = [np.asarray(x.todense()) for x in X_test]
a_val = [np.asarray(a.todense()) for a in A_test]
# x_y = [np.asarray(x_y.todense()) for x_y in X_y_train]
# y = [np.asarray(y.todense()) for y in y_train]

tr = [np.stack(x_t, axis=0), np.stack(a, axis=0)]
val = [np.stack(x_t_val, axis=0), np.stack(a_val, axis=0)]
# yd = np.concatenate((np.stack(x_y, axis=0), np.stack(y, axis=0)), axis=2)
yd = y_train
yd_val = y_test

preds = None

history = model.fit(tr, yd, epochs=NB_EPOCH, verbose=2, batch_size=1, validation_data=(val, yd_val))
preds = model.predict(tr, batch_size=1)

plt.subplot(3, 1, 1)
plt.matshow(tr[0][0:40,0:28],cmap=plt.cm.Blues)
plt.subplot(3, 1, 2)
plt.matshow(yd[0:40,0:28],cmap=plt.cm.Blues)
plt.subplot(3, 1, 3)
plt.matshow(preds[0:40,0:28],cmap=plt.cm.Blues)
plt.show()
# Fit
for epoch in range(1, NB_EPOCH + 1):

    # Log wall-clock time
    t = time.time()

    # Single training iteration
    model.fit(tr, yd, batch_size=128, nb_epoch=1, shuffle=False, verbose=0)

    if epoch % 1 == 0:

        # Predict on full dataset
        preds = model.predict(tr, batch_size=128)

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [idx_train, idx_val])

        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

    else:
        print("Epoch: {:04d}".format(epoch),
              "time= {:.4f}".format(time.time() - t))

# Testing
test_loss, test_acc = evaluate_preds(preds, [y_test], [idx_test])
print("Test set results:",
      "loss= {:.4f}".format(test_loss[0]),
      "accuracy= {:.4f}".format(test_acc[0]))