import numpy as np

# Функция активации - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

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

# Обучение перцептрона
learning_rate = 0.5
epochs = 10000

for _ in range(epochs):
    # Прямое распространение
    input_layer = X
    weighted_sum = np.dot(input_layer, weights) + bias
    activated_output = sigmoid(weighted_sum)

    # Ошибка и коррекция весов
    error = y - activated_output
    adjustment = error * sigmoid_derivative(activated_output)
    weights += np.dot(input_layer.T, adjustment) * learning_rate
    bias += np.sum(adjustment) * learning_rate

# Предсказание
input_data = np.array([1, 0])
result = sigmoid(np.dot(input_data, weights) + bias)
print("Предсказание для [1, 0]:", result)