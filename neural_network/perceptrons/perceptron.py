import numpy as np


def test_perceptron():
    global w
    examples = np.array([[1, 1, 0.3], [1, 0.4, 0.5], [1, 0.7, 0.8]])
    w = np.array([0, 0, 0])

    def Predict(example):
        if example[1] == 1 and example[2] == 0.3:
            return 1
        elif example[1] == 0.4 and example[2] == 0.5:
            return 1
        elif example[1] == 0.7 and example[2] == 0.8:
            return 0

    def Target(example):
        sum = example @ w
        print(sum)
        if sum > 0:
            return 1
        else:
            return 0

    perfect = False
    while not perfect:
        perfect = True
        for e in examples:
            if Predict(e) != Target(e):
                perfect = False
                if Predict(e) == 1:
                    w = w + e
                elif Predict(e) == 0:
                    w = w - e
        break
    print(w)

class Perceptron():
    def __init__(self, n = 1, bias = 0):
        self.weights = np.zeros(n)
        self.bias = bias

    def predict(self, x):
        return (self.weights * x).sum() + self.bias > 0

    def iter_learn(self, x, answer):
        prediction = self.predict(x)
        if prediction != answer:
            if prediction:
                self.weights -= x
            else:
                self.weights += x

perceptron = Perceptron(2)
perceptron.iter_learn(np.array([1.0, 0.3]), True)
perceptron.iter_learn(np.array([0.4, 0.5]), True)
perceptron.iter_learn(np.array([0.7, 0.8]), False)

print(perceptron.bias, perceptron.weights)