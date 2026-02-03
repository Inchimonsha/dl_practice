# для красоты
# можете закомментировать, если у вас не установлен этот пакет
import seaborn

import matplotlib.pyplot as plt
import numpy as np

# логистическая функция
def logit(x):
    return 1 / (1 + np.exp(-x))

# наша линия: w1 * x1 + w2 * x2 + b = 0
def line(x1, x2):
    return -3 * x1 - 5 * x2 - 2


# служебная функция в форме x2 = f(x1) (для наглядности)
def line_x1(x1):
    return (-3 * x1 - 2) / 5


# генерируем диапазон точек
np.random.seed(0)
x1x2 = np.random.randn(200, 2) * 2

# рисуем точки
for x1, x2 in x1x2:
    value = logit(line(x1, x2) / 2)

    if value < 0.001:
        color = "red"
    elif value > 0.999:
        color = "green"
    else:
        color = str(0.75 - value * 0.5)


    if (value == 0):  # синие — на линии
        plt.plot(x1, x2, 'ro', color=color)
    elif (value > 0):  # зелёные — выше линии
        plt.plot(x1, x2, 'ro', color=color)
    elif (value < 0):  # красные — ниже линии
        plt.plot(x1, x2, 'ro', color=color)

# выставляем равное пиксельное  разрешение по осям
plt.gca().set_aspect('equal', adjustable='box')

# рисуем саму линию
x1_range = np.arange(-5.0, 5.0, 0.5)
plt.plot(x1_range, line_x1(x1_range), color='blue')

# проставляем названия осей
plt.xlabel('x1')
plt.ylabel('x2')

# на экран!
plt.show()