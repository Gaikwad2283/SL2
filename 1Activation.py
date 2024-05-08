import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Define input range
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')

plt.subplot(2, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')

# Softmax requires special handling due to multi-class output
x_softmax = np.array([2.0, 1.0, 0.1])
plt.subplot(2, 3, 5)
plt.bar(range(len(x_softmax)), softmax(x_softmax))
plt.title('Softmax')

plt.tight_layout()
plt.show()
