import numpy as np

# Define the binary representations of digits 0, 1, 2, 3, and 9
digit_0 = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
])

digit_1 = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 1]
])

digit_2 = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 1]
])

digit_3 = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

digit_9 = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])

# Flatten the digits for training
X_train = np.array([digit_0.flatten(), digit_1.flatten(), digit_2.flatten(), digit_3.flatten(), digit_9.flatten()])
y_train = np.array([[1, 0, 0, 0, 0],   # 0
                    [0, 1, 0, 0, 0],   # 1
                    [0, 0, 1, 0, 0],   # 2
                    [0, 0, 0, 1, 0],   # 3
                    [0, 0, 0, 0, 1]])  # 9

# Define the neural network architecture
input_size = 15
hidden_size = 10
output_size = 5
learning_rate = 0.1

# Initialize weights
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network
for epoch in range(10000):
    # Forward pass
    layer1 = sigmoid(np.dot(X_train, W1))
    output = sigmoid(np.dot(layer1, W2))
    
    # Backpropagation
    error = y_train - output
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(layer1)
    
    # Update weights
    W2 += layer1.T.dot(d_output) * learning_rate
    W1 += X_train.T.dot(d_hidden) * learning_rate

# Test the neural network
def predict_digit(digit):
    layer1 = sigmoid(np.dot(digit.flatten(), W1))
    output = sigmoid(np.dot(layer1, W2))
    return np.argmax(output)

# Test data
test_digit = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 1]
])

prediction = predict_digit(test_digit)
print("Predicted digit:", prediction)