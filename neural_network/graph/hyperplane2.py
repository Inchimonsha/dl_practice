# для красоты
# можете закомментировать, если у вас не установлен этот пакет
import seaborn

import matplotlib.pyplot as plt
import numpy as np


# образец 1
def line1(w1, w2):
    return -3 * w1 - 5 * w2 - 8


# служебная функция в форме w2 = f1(w1) (для наглядности)
def line1_w1(w1):
    return (-3 * w1 - 8) / 5


# образец 2
def line2(w1, w2):
    return 2 * w1 - 3 * w2 + 4


# служебная функция в форме w2 = f2(w1) (для наглядности)
def line2_w1(w1):
    return (2 * w1 + 4) / 3


# образец 3
def line3(w1, w2):
    return 1.2 * w1 - 3 * w2 + 4


# служебная функция в форме w2 = f2(w1) (для наглядности)
def line3_w1(w1):
    return (1.2 * w1 + 4) / 3


# образец 4
def line4(w1, w2):
    return -5 * w1 - 5 * w2 - 8


# служебная функция в форме w2 = f2(w1) (для наглядности)
def line4_w1(w1):
    return (-5 * w1 - 8) / 5


# генерируем диапазон точек
w1_range = np.arange(-5.0, 5.0, 0.5)
w2_range = np.arange(-5.0, 5.0, 0.5)

# рисуем веса (w1, w2), лежащие по нужные стороны от образцов
for w1 in w1_range:
    for w2 in w2_range:
        value1 = line1(w1, w2)
        value2 = line2(w1, w2)
        value3 = line3(w1, w2)
        value4 = line4(w1, w2)

        if (value1 < 0 and value2 > 0 and value3 > 0 and value4 < 0):
            color = 'green'
        else:
            color = 'pink'

        plt.plot(w1, w2, 'ro', color=color)

# выставляем равное пиксельное  разрешение по осям
plt.gca().set_aspect('equal', adjustable='box')

# рисуем саму линию (гиперплоскость) для образца 1
plt.plot(w1_range, line1_w1(w1_range), color='blue')
# для образца 2
plt.plot(w1_range, line2_w1(w1_range), color='blue')
# для образца 3
plt.plot(w1_range, line3_w1(w1_range), color='blue')
# для образца 4
plt.plot(w1_range, line4_w1(w1_range), color='blue')

# рисуем только эту область — остальное не интересно
plt.axis([-7, 7, -7, 7])

# проставляем названия осей
plt.xlabel('w1')
plt.ylabel('w2')

# на экран!
plt.show()