#-------------------------------------------------------------------------------
#STEP 01: TẠO DỮ LIỆU GIẢ
#region TẠO DỮ LIỆU GIẢ
import math
import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
d0 = 2 # dimensionality
C = 3 # number of classes
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels

for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j

# lets visualize the data:
plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7);

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

#plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
plt.show()
#endregion

#-------------------------------------------------------------------------------
#STEP 02: ĐỊNH NGHĨA CÁC HÀM BỔ TRỢ
#region ĐỊNH NGHĨA CÁC HÀM BỔ TRỢ
## Softmax function
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
def cost(Y, Yhat, W1, W2, W3):
    w1 = np.square(W1).sum()
    w2 = np.square(W2).sum()
    w3 = np.square(W3).sum()
    R = w1 + w2 + w3
    return (-np.sum(Y*np.log(Yhat))/Y.shape[1])+(l/(2*Y.shape[1]))*R
#endregion

#-------------------------------------------------------------------------------
#STEP 03: HUẤN LUYỆN MẠNG
#region HUẤN LUYỆN MẠNG
d0 = 2
d1 = h = 100 # size of hidden layer
d2 = 20
d3 = C = 3
# initialize parameters randomly
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))
W3 = 0.01*np.random.randn(d2, d3)
b3 = np.zeros((d3, 1))

Y = convert_labels(y, C)

#print(Y)
N = X.shape[1]
eta = 1 # learning rate
for i in range(1001):
    ## Feedforward
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0) #ReLU
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0) #ReLU
    Z3 = np.dot(W3.T, A2) + b3

    Yhat = softmax(Z3) #Softmax for output

    # print loss after each 1000 iterations
    if i %10 == 0:
        # compute the loss: average cross-entropy loss
        loss = cost(Y, Yhat, W1, W2, W3)
        print("iter %d, loss: %f" %(i, loss))

    # backpropagation
    
    E3 = (Yhat - Y )/N
    dW3 = np.dot(A2, E3.T)
    db3 = np.sum(E3, axis = 1, keepdims = True)
    E2 = np.dot(W3, E3)
    E2[Z2 <= 0] = 0 # gradient of ReLU
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis = 1, keepdims = True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # gradient of ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis = 1, keepdims = True)

    # Gradient Descent update
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2
    W3 += -eta*dW3
    b3 += -eta*db3
#endregion

# STEP 04: ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MẠNG SAU KHI HUẤN LUYỆN
#region ĐÁNH GIÁ ĐỘ CHÍNH XÁC CỦA MẠNG SAU KHI HUẤN LUYỆN
Z1 = np.dot(W1.T, X) + b1 
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
A2 = np.maximum(Z2, 0)
Z3 = np.dot(W3.T, A2) + b3

predicted_class = np.argmax(Z3, axis=0)
acc = (100*np.mean(predicted_class == y))
print('training accuracy: %.2f %%' % acc)
#endregion

#-------------------------------------------------------------------------------
# STEP 05: TRỰC QUAN HÓA KẾT QUẢ PHAN LOẠI BẰNG MẠNG
#region TRỰC QUAN HÓA KẾT QUẢ PHAN LOẠI BẰNG MẠNG

# Visualize results
#Visualize 
xm = np.arange(-1.5, 1.5, 0.025)
xlen = len(xm)
ym = np.arange(-1.5, 1.5, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)

print(np.ones((1, xx.size)).shape)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

X0 = np.vstack((xx1, yy1))

Z1 = np.dot(W1.T, X0) + b1 
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
# predicted class 
Z = np.argmax(Z2, axis=0)

Z = Z.reshape(xx.shape)
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)

# X = X.T
N = 100
plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7);
plt.plot(X[0, N:2*N], X[1, N:2*N], 'g^', markersize = 7);
plt.plot(X[0, 2*N:], X[1, 2*N:], 'ro', markersize = 7);

plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xticks(())
plt.yticks(())
plt.title('#hidden units = %d, accuracy = %.2f %%' %(d1, acc))
plt.show()
#endregion

