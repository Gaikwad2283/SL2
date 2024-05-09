import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

num_0 = np.array([[1,1,1],
                 [1,0,1],
                 [1,0,1],
                 [1,0,1],
                 [1,1,1]])
num_1 = np.array([[0,1,0],
                 [0,1,0],
                 [0,1,0],
                 [0,1,0],
                 [1,1,1]])
num_2 = np.array([[1,1,1],
                  [0,0,1],
                  [1,1,1],
                  [1,0,0],
                  [1,1,1]])
num_3 = np.array([[1,1,1],
                 [0,0,1],
                 [1,1,1],
                 [0,0,1],
                 [1,1,1]])
num_9 = np.array([[1,1,1],
                 [1,0,1],
                 [1,1,1],
                 [0,0,1],
                 [1,1,1]])


X = np.array([num_0.flatten(), num_1.flatten(), num_2.flatten(), num_3.flatten(), num_9.flatten()])
y = np.array([[0,0,0,0,0],
             [0,0,0,0,1],
              [0,0,0,1,0],
              [0,0,0,1,1],
              [1,0,0,0,1]])
weights = np.random.randn(15,5)
bias = np.random.randn(1)
lr = 0.5
print(X)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def costfn(actual, pred):
    return np.mean((actual-pred)**2)


def forward(x, weights, bias):
    z = np.dot(x, weights) + bias
    return sigmoid(z)

def backward(x, y, output, weights, bias, lr):
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_bias = np.sum(d_output)
    d_weights = np.dot(x.T, d_output)
    
    weights += lr * d_weights
    bias += lr * d_bias
    
    return weights, bias


for epoch in range(10000):
    output = forward(X, weights, bias)
    weights, bias = backward(X,y, output, weights, bias, lr)
    
    cost = costfn(y, output)
    if epoch%1000 == 0:
        print(f"epoch {epoch} : ", cost)
        
        test = np.array([[1,1,1],
                 [1,0,1],
                 [1,1,1],
                 [0,0,1],
                 [1,1,1]])
forward(test.flatten(), weights, bias).round()