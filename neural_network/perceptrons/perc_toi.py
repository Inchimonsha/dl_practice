import numpy as np

class Perceptron:
    def __init__(self, n = 1, bias = 0, lr=0.1):
        self.weights = np.zeros(n)
        self.bias = bias
        self.lr = lr

    def predict(self, x):
        return (self.weights * x).sum() + self.bias > 0

    def mse(self, x, y):
        return (y - self.predict(x))**2

    def learn(self, x, answer):
        prediction = self.predict(x)
        if prediction != answer:
            if prediction:
                self.weights -= x * self.lr
            else:
                self.weights += x * self.lr

    def train(self, x_train, y_train, epochs=10):
        for _ in range(epochs):
            for i in range(len(x_train)):
                self.learn(x_train[i], y_train[i])


x_train = [np.array([1, 1]), np.array([1, 0]), np.array([0, 1]), np.array([0, 0])]
y_train = [1, 1, 1, 0]

perceptron = Perceptron(2)
perceptron.train(x_train, y_train)

predictions = [perceptron.predict(x) for x in x_train]
mse_values = [perceptron.mse(x, y) for x, y in zip(x_train, y_train)]

print("Model Bias:", perceptron.bias)
print("Model Weights:", perceptron.weights)
print("Predictions:", predictions)
print("MSE Values:", mse_values)