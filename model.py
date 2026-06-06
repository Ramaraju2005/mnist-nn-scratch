import numpy as np


class NeuralNetwork:
    """
    A simple 3-layer fully connected neural network built from scratch using NumPy.
    Architecture: Input(784) -> Hidden(128) -> Output(10)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation

        # He initialization for weights (good for ReLU)
        self.params = {
            'W1': np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size),
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size),
            'b2': np.zeros((output_size, 1)),
        }

        self.cache = {}
        self.grads = {}
        self.loss_history = []
        self.accuracy_history = []

    # ---------- Activation Functions ----------

    def relu(self, z, derivative=False):
        if derivative:
            return (z > 0).astype(float)
        return np.maximum(0, z)

    def sigmoid(self, z, derivative=False):
        sig = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        if derivative:
            return sig * (1 - sig)
        return sig

    def activate(self, z, derivative=False):
        if self.activation_name == 'relu':
            return self.relu(z, derivative)
        return self.sigmoid(z, derivative)

    def softmax(self, z):
        # Numerically stable softmax
        shifted = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # ---------- Forward Pass ----------

    def forward(self, X):
        """
        X shape: (batch_size, 784)
        """
        self.cache['X'] = X

        # Layer 1: Linear + Activation
        self.cache['Z1'] = self.params['W1'] @ X.T + self.params['b1']  # (128, batch)
        self.cache['A1'] = self.activate(self.cache['Z1'])               # (128, batch)

        # Layer 2: Linear + Softmax
        self.cache['Z2'] = self.params['W2'] @ self.cache['A1'] + self.params['b2']  # (10, batch)
        self.cache['A2'] = self.softmax(self.cache['Z2'])                              # (10, batch)

        return self.cache['A2']

    # ---------- Loss ----------

    def cross_entropy_loss(self, y_pred, y_true):
        """
        y_pred: (10, batch_size) probabilities
        y_true: (batch_size, 10) one-hot labels
        """
        batch_size = y_true.shape[0]
        # Clip to avoid log(0)
        log_probs = -np.log(y_pred.T + 1e-9)
        loss = np.sum(y_true * log_probs) / batch_size
        return loss

    # ---------- Backward Pass ----------

    def backward(self, y_true):
        """
        y_true: (batch_size, 10) one-hot labels
        """
        batch_size = y_true.shape[0]

        # Gradient of loss w.r.t Z2 (softmax + cross-entropy combined derivative)
        dZ2 = self.cache['A2'] - y_true.T                             # (10, batch)
        dW2 = (1 / batch_size) * dZ2 @ self.cache['A1'].T             # (10, 128)
        db2 = (1 / batch_size) * np.sum(dZ2, axis=1, keepdims=True)   # (10, 1)

        # Gradient flowing back to hidden layer
        dA1 = self.params['W2'].T @ dZ2                                # (128, batch)
        dZ1 = dA1 * self.activate(self.cache['Z1'], derivative=True)  # (128, batch)
        dW1 = (1 / batch_size) * dZ1 @ self.cache['X']                # (128, 784)
        db1 = (1 / batch_size) * np.sum(dZ1, axis=1, keepdims=True)   # (128, 1)

        self.grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        return self.grads

    # ---------- Optimizer ----------

    def update_params(self, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * self.grads[key]

    # ---------- Accuracy ----------

    def accuracy(self, y_pred, y_true):
        predicted = np.argmax(y_pred.T, axis=1)
        actual = np.argmax(y_true, axis=1)
        return np.mean(predicted == actual)

    # ---------- Predict ----------

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output.T, axis=1)
