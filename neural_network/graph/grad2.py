import time

import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return x**2 + x + 1

def derivative(func, x, n, h = 1e-3):
    if n == 0:
        return func(x)
    else:
        return ((derivative(func, x + h, n - 1, h) -
                derivative(func, x - h, n - 1, h))
                / (2 * h))


if __name__ == "__main__":
    x_range = np.linspace(-10, 10, 50)
    plt.ion()
    fig, ax = plt.subplots()
    ax.grid(True)

    x_prev = -5
    ax.plot(x_range, f(x_range))
    point = ax.scatter(x_prev, f(x_prev), color="red")
    lr = 0.8
    accuracy = 1e-3
    while True:
        x = x_prev - lr * derivative(f, x_prev, 1)

        point.set_offsets([x, f(x)])
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.02)
        if abs(f(x) - f(x_prev)) < accuracy:
            break
        x_prev = x

    plt.ioff()
    print(x)
    ax.scatter(x, f(x), color="blue")
    plt.show()