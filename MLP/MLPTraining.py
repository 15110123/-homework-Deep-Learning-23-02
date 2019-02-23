import numpy as np
import MLPFunctions as mLPFunctions
import matplotlib.pyplot as plt

def train(X, y, layerCount, nodeCounts):
    d0 = 2
    # LayerCount là hidden layer count. Ta gọi dl là d của lớp cuối (output layer) 
    dl = C = 3
    # Tạo một danh sách w, sẽ kích thước là layerCount
    w = []
    b = []
    W1 = 0.01*np.random.randn(d0, nodeCounts[0])
    b1 = np.zeros((nodeCounts[0], 1))

    i = 0
    while i < layerCount - 1:
        w.append(0.01*np.random.randn(nodeCounts[i], nodeCounts[i + 1]))
        b.append(np.zeros((nodeCounts[i + 1], 1)))
        i = i + 1

    Wl = 0.01*np.random.randn(nodeCounts[layerCount - 1], dl)
    bl = np.zeros((dl, 1))

    Y = mLPFunctions.convert_labels(y, C)

    #print(Y)
    N = X.shape[1]
    eta = 1 # learning rate
    for i in range(1000):
        ## Feedforward
        Z1 = np.dot(W1.T, X) + b1
        A1 = np.maximum(Z1, 0) #ReLU

        Z = []
        A = []
        
        i = 0
        while i < layerCount:
            preA = A1
            if i != 0:
                preA = A[i - 1]
            Z.append(np.dot(w[i].T, preA) + b[i])
            A.append(np.maximum(Z[i], 0))

        Zl = np.dot(wl.T, A[layerCount - 1]) + bl

        Yhat = mLPFunctions.softmax(Zl) #Softmax for output

        # print loss after each 1000 iterations
        if i %10 == 0:
            # compute the loss: average cross-entropy loss
            loss = mLPFunctions.cost(Y, Yhat, W1, W2, W3)
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

    Z1 = np.dot(W1.T, X) + b1 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = np.dot(W3.T, A2) + b3

    predicted_class = np.argmax(Z3, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.2f %%' % acc)

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