import numpy as np


# Сигмоидная функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MSE(I, a, n):
    '''
    Mean Squared Error
    Среднеквадратичная ошибка
    :param I: expected result \\ ожидаемый результат
    :param a: obtained result \\ полученный результат
    :param n: number of sets \\ количество сетов
    :return: error of discrepancy between expected and received responses \\
    ошибка расхождения между ожидаемым и полученным ответами
    '''
    rsum = 0
    for i in range(n):
        rsum += (I[i] - a[i])**2
    return rsum / n

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])  # Веса
        bias = 0  # Смещение
        self.h1 = Neuron(np.array([0.45, -0.12]), bias)
        self.h2 = Neuron(np.array([0.78, 0.13]), bias)
        self.o1 = Neuron(np.array([1.5, -2.3]), bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = NeuralNetwork()
x = np.array([1, 0])  # Входные значения
output = network.feedforward(x)
error = MSE([1], [output], 1)
print(output, error)