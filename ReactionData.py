import numpy as np
import scipy.sparse as sp
import keras

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, A, X, y, X_y, batch_size=32):
        self.A = A
        self.X = X
        self.y = y
        self.X_y = X_y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.A)/float(self.batch_size)))

    def __iter__(self):
        for a, x, y, x_y in zip(self.A, self.X, self.y, self.X_y):
            yield np.hstack((x, a.todense*())), np.hstack((x_y, y.todense()))

    def __getitem__(self, idx):
        a_idx = self.A[idx*self.batch_size:(idx + 1)*self.batch_size]
        x_idx = self.X[idx*self.batch_size:(idx + 1)*self.batch_size]
        y_idx = self.y[idx*self.batch_size:(idx + 1)*self.batch_size]
        x_y_idx = self.X_y[idx*self.batch_size:(idx + 1)*self.batch_size]
        #a_idx = [np.asarray(a.todense()) for a in a_idx]
        #x_idx = [np.asarray(x.todense()) for x in x_idx]
        #y_idx = [np.asarray(y.todense()) for y in y_idx]
        #x_y_idx = [np.asarray(x_y.todense()) for x_y in x_y_idx]
        #a_idx = np.stack(a_idx, axis=0)
        #x_idx = np.stack(x_idx, axis=0)
        #y_idx = np.stack(y_idx, axis=0)
        #x_y_idx = np.stack(x_y_idx, axis=0)
        #batch_x = [sp.hstack((x, a.todense())) for a, x in zip(a_idx, x_idx)]
        #batch_y = [sp.hstack((x_y, y.todense())) for y, x_y in zip(y_idx, x_y_idx)]
        return np.concatenate((x_idx[0].todense(), a_idx[0].todense()), axis=1), np.concatenate((x_y_idx[0].todense(), y_idx[0].todense()), axis=1)