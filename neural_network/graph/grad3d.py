import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def E(y, a, b):
    ff = np.array([a * x + b for x in range(N)])
    return np.dot((y - ff).T, (y - ff))

def dEda(y, a, b):
    ff = np.array([a * x + b for x in range(N)])
    return -2 * np.dot((y - ff).T, range(N))

def dEdb(y, a, b):
    ff = np.array([a * x + b for x in range(N)])
    return -2 * (y - ff).sum()

def derivative(func, y, a, b, n, h = 1e-3):
    if n == 0:
        return func(y, a, b)
    else:
        return ((derivative(func, y, a + h, b + h, n - 1, h) -
                derivative(func, y, a - h, b - h, n - 1, h))
                / (2 * h))


if __name__ == "__main__":
    N = 100
    sigma = 3
    at = 0.5
    bt = 2

    aa = 0
    bb = 0

    f = np.array([at * x + bt for x in range(N)])
    y = np.array(f + np.random.normal(0, sigma, N))

    a_plt = np.arange(-1, 2, 0.1)
    b_plt = np.arange(0, 3, 0.1)
    E_plt = np.array([[E(y, a, b) for a in a_plt] for b in b_plt])

    plt.ion()
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    a, b = np.meshgrid(a_plt, b_plt)
    axes.plot_surface(a, b, E_plt, color="blue", alpha=0.5)

    axes.set_xlabel("a")
    axes.set_ylabel("b")
    axes.set_zlabel("E")

    point = axes.scatter(aa, bb, E(y, aa, bb), color="red")

    for i in range(100):
        aa = aa - 0.000001 * dEda(y, aa, bb)
        bb = bb - 0.0005 * dEdb(y, aa, bb)

        axes.scatter(aa, bb, E(y, aa, bb), color="red")

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

        print(aa, bb)

    plt.ioff()
    plt.show()