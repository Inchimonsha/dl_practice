import numpy as np

class Perceptron:
    def __init__(self, weights, bias, learning_rate=0.1, epochs=10):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Функция активации - sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Производная sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def predict(self, input_layer):
        weighted_sum = np.dot(input_layer, self.weights) + self.bias
        activated_output = self.sigmoid(weighted_sum)
        return activated_output

    def mse(self, x, y):
        return (y - x)**2

    def learn(self, input_layer, answer):
        activated_output = self.predict(input_layer)
        # Ошибка и коррекция весов
        error = self.mse(activated_output, answer)
        adjustment = error * self.sigmoid_derivative(activated_output)
        self.weights = self.weights + input_layer.T * adjustment * self.learning_rate
        self.bias += np.sum(adjustment) * self.learning_rate

    def train(self, x_train, y_train):
        for _ in range(self.epochs):
            for i in range(len(x_train)):
                self.learn(x_train[i], y_train[i])


# Набор данных
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Ожидаемые результаты
y = np.array([[0],
              [1],
              [1],
              [0]])

# Инициализация весов и смещения
np.random.seed(1)
weights = 2 * np.random.random((2, 1)) - 1
bias = 2 * np.random.random() - 1
learning_rate = 0.5
epochs = 10

perceptron = Perceptron(weights, bias, learning_rate, epochs)
perceptron.train(X, y)

predictions = [perceptron.predict(x) for x in X]
mse_values = [perceptron.mse(x, y) for x, y in zip(X, y)]

print("Model Bias:", perceptron.bias)
print("Model Weights:", perceptron.weights)
print("Predictions:", predictions)
print("MSE Values:", mse_values)

print(perceptron.predict([1, 0]))
print(perceptron.predict([1, 1]))
print(perceptron.predict([0, 0]))