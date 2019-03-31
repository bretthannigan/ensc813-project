from __future__ import print_function

import os
import sys
sys.path.append(os.path.abspath('./relational-gcn/'))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import pickle as pkl

import time

from ReactionData import DataGenerator

np.random.seed()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="LoweUSPTOGrants_1976-2016_128Atoms_1000Reactions.pickle",
                help="Dataset string ('aifb', 'mutag', 'bgs', 'am')")
ap.add_argument("-e", "--epochs", type=int, default=500,
                help="Number training epochs")
ap.add_argument("-hd", "--hidden", type=int, default=256,
                help="Number hidden units")
ap.add_argument("-do", "--dropout", type=float, default=0.,
                help="Dropout rate")
ap.add_argument("-b", "--bases", type=int, default=-1,
                help="Number of bases used (-1: all)")
ap.add_argument("-lr", "--learnrate", type=float, default=0.01,
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

with open(DATASET, 'rb') as f:
    reactant = pkl.load(f)
    product = pkl.load(f)

A = [r.get_adjacency(normalize=True) for r in reactant]
X = [r.get_features() for r in reactant]
y = [p.get_adjacency(normalize=True) for p in product]
X_y = [p.get_features() for p in product]

A_train, A_test, X_train, X_test, y_train, y_test, X_y_train, X_y_test = train_test_split(A, X, y, X_y, train_size=TRAIN_FRACTION, random_state=0, shuffle=True)

num_nodes = A[0].shape[0]
support = 1

A_in = [InputAdj(sparse=True)]

training_data_generator = DataGenerator(A_train, X_train, y_train, X_y_train, batch_size=128)
validation_data_generator = DataGenerator(A_test, X_test, y_test, X_y_test, batch_size=128)

# Define model architecture
X_in = Input(shape=(X[0].shape[1],))
H = Dropout(DO)(X_in)
# H = GraphConvolution(1024, support, num_bases=BASES, featureless=False, bias=True,
#                      activation='relu')([H] + A_in)#,
#                      #W_regularizer=l1(L2))([H] + A_in)
#H = Dropout(DO)(H)
H = GraphConvolution(10000, support, num_bases=BASES, featureless=False, bias=True,
                     activation='relu')([H] + A_in)
# H = GraphConvolution(1024, support, num_bases=BASES, featureless=False, bias=True,
#                      activation='relu')([H] + A_in)
# H = GraphConvolution(1024, support, num_bases=BASES, featureless=False, bias=True,
#                      activation='relu')([H] + A_in)
# #H = Dropout(DO)(H)
# H = GraphConvolution(1024, support, num_bases=BASES, featureless=False, bias=True,
#                      activation='relu')([H] + A_in)
#H = Dropout(DO)(H)
Y = GraphConvolution(y_train[0].shape[1] + X_y_train[0].shape[1], support, num_bases=BASES,
                     activation='linear')([H] + A_in)

# Compile model
model = Model(inputs=[X_in] + A_in, outputs=Y)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR))
model.summary()
tr = [sp.vstack(X_train).todense(), sp.vstack(A_train)]
yd = sp.vstack(y_train)
tr = [X_train[0].todense(), A_train[0]]
yd = np.concatenate((X_y_train[0].todense(), y_train[0].todense()), axis=1)
#tr = sp.vstack(([sp.hstack((x.todense(), a)) for x, a in zip(X_train, A_train)]))
preds = None
#model.fit([np.concatenate((X_train[0].todense(), X_train[1].todense()), axis=0), sp.vstack((A_train[0], A_train[1]))], sp.vstack((y_train[0], y_train[1])), epochs=50, verbose=2, batch_size=128)
model.fit(tr, yd, epochs=NB_EPOCH, verbose=2, batch_size=1)
preds = model.predict(tr, batch_size=128)

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