import numpy as np
import MLPFunctions as mLPFunctions
import matplotlib.pyplot as plt

def train(X, y, layerCount, nodeCounts):
    d0 = 2
    # LayerCount là hidden layer count. Ta gọi dl là d của lớp cuối (output layer) 
    dl = C = 3
    # Tạo một danh sách w, sẽ kích thước là layerCount
    W = []
    b = []

    i = 0
    while i <= layerCount - 1:
        preNodeCount = d0
        if i != 0:
            preNodeCount = nodeCounts[i - 1]
        W.append(0.01*np.random.randn(preNodeCount, nodeCounts[i]))
        b.append(np.zeros((nodeCounts[i], 1)))
        i = i + 1

    Wl = 0.01*np.random.randn(nodeCounts[layerCount - 1], dl)
    bl = np.zeros((dl, 1))

    Y = mLPFunctions.convert_labels(y, C)

    N = X.shape[1]
    eta = 1 # Tham chiếu learning rate
    for j in range(1000):
        ## Feedforward
        Z = []
        A = []
        
        i = 0
        while i < layerCount:
            preA = X
            if i != 0:
                preA = A[i - 1]
            Z.append(np.dot(W[i].T, preA) + b[i])
            A.append(np.maximum(Z[i], 0))
            i = i + 1

        Zl = np.dot(Wl.T, A[layerCount - 1]) + bl

        Yhat = mLPFunctions.softmax(Zl) #Softmax

        # In loss sau mỗi 1000 vòng lặp 
        if j %10 == 0:
            # average cross-entropy loss
            loss = mLPFunctions.cost(Y, Yhat, Wl, W)
            print("iter %d, loss: %f" %(j, loss))

        # backpropagation
        El = (Yhat - Y )/N
        dWl = np.dot(A[layerCount - 1], El.T)
        dbl = np.sum(El, axis = 1, keepdims = True)

        EPre = El
        dW = []
        db = []

        i = layerCount - 1
        while i >= 0:
            Wt = Wl
            if i < layerCount - 1:
                Wt = W[i + 1]
            # EPre là E tính trước đó, EPre cập nhật sau mỗi vòng lặp 
            EPre = np.dot(Wt, EPre)
            EPre[Z[i] <= 0] == 0
            if i != 0:
                dW.append(np.dot(A[i - 1], EPre.T))
            else:
                dW.append(np.dot(X, EPre.T))
            db.append(np.sum(EPre, axis = 1, keepdims = True))
            i = i - 1

        # Đảo ngược mảng dW, do vòng lặp trước đi lùi về 0
        dW.reverse()
        db.reverse()

        # Cập nhật gradient Descent

        for i in range(0, layerCount - 1):
            W[i] += -eta*dW[i]
            b[i] += -eta*db[i]
        
        Wl += -eta*dWl
        bl += -eta*dbl

    # Apre là A trước đó, tại i - 1 hoặc A1 (i == 0)
    Apre = X

    for i in range(0, layerCount - 1):
        Z[i] = np.dot(W[i].T, Apre) + b[i]
        A[i] = np.maximum(Z[i], 0)
        Apre = A[i]

    Zl = np.dot(Wl.T, A[layerCount - 1]) + bl

    predicted_class = np.argmax(Zl, axis=0)
    acc = (100*np.mean(predicted_class == y))
    print('training accuracy: %.2f %%' % acc)

    # Trực quan hóa kết quả phân loại 
    xm = np.arange(-1.5, 1.5, 0.025)
    xlen = len(xm)
    ym = np.arange(-1.5, 1.5, 0.025)
    ylen = len(ym)
    xx, yy = np.meshgrid(xm, ym)

    print(np.ones((1, xx.size)).shape)
    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    X0 = np.vstack((xx1, yy1))

    Z1 = np.dot(W[0].T, X0) + b[0] 
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W[1].T, A1) + b[1]
    # predicted class 
    Zm = np.argmax(Z2, axis=0)

    Zm = Zm.reshape(xx.shape)
    CS = plt.contourf(xx, yy, Zm, 200, cmap='jet', alpha = .1)

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
    plt.title('#hidden units = %d, accuracy = %.2f %%' %(nodeCounts[0], acc))
    plt.show()