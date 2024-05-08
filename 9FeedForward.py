import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt


X  = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
             [1],
             [1],
             [1]])
lr = 0.5
weights = np.random.randn(2,1)
bias = np.random.randn(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def costfn(actual, pred):
    return np.mean((actual-pred)**2)

def forward(x, weights, bias):
    z = np.dot(x, weights) + bias
    return sigmoid(z)

def backward(x, y, output, bias, weights, lr):
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_bias = np.sum(d_output)
    d_weights = np.dot(x.T, d_output)
    
    weights += lr * d_weights
    bias += lr * d_bias
    
    return weights, bias


for epoch in range(30000):
    output = forward(X, weights, bias)
    weights, bias = backward(X, y, output, bias, weights, lr)
    
    cost = costfn(y, output)
    if epoch%5000 == 0:
        print(f"epoch {epoch} : ", cost)
        
        test = np.array([0,1])
forward(test, weights, bias).round()