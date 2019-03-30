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
            yield np.hstack((a, x)), np.hstack((y, x_y))

    def __getitem__(self, idx):
        a_idx = self.A[idx*self.batch_size:(idx + 1)*self.batch_size]
        x_idx = self.X[idx*self.batch_size:(idx + 1)*self.batch_size]
        y_idx = self.y[idx*self.batch_size:(idx + 1)*self.batch_size]
        x_y_idx = self.X_y[idx*self.batch_size:(idx + 1)*self.batch_size]
        batch_x = [sp.hstack((a, x)) for a, x in zip(a_idx, x_idx)]
        batch_y = [sp.hstack((y, x_y)) for y, x_y in zip(y_idx, x_y_idx)]
        return batch_x, batch_y