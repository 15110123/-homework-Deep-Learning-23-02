import numpy as np

def softmax(V):
    e_V = np.exp(V - np.max(V, axis = 0, keepdims = True))
    Z = e_V / e_V.sum(axis = 0)
    return Z

## One-hot coding
from scipy import sparse
def convert_labels(y, C = 3):
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

# cost or loss function
l = 0.01
def cost(Y, Yhat, W1, Wl, W):
    w = []
    w1 = np.square(W1).sum()
    for i in W:
        w.append(np.square(i).sum())
    wl = np.square(Wl).sum()
    R = w1 + sum(w) + wl
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R
