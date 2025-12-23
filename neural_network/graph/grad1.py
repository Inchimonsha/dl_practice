import numpy as np
from matplotlib import pyplot as plt

X = np.linspace(-6, 6, 20)/6

w_0, w_1 = 4, -2
#истинная функция
y = lambda x: w_0 + w_1 * x

w0 = np.linspace(-2, 10, 10)
w1 = np.linspace(-8, 4, 10)
W0, W1 = np.meshgrid(w0, w1)
#предсказанная функция
y_predict = lambda x: W0 + W1*x

# целевая функция
J = sum([(y_predict(x) - y(x))**2 for x in X])

# График целевой функции
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(W0, W1, J, color='green')
ax.plot_surface(W0, W1, J, cmap='inferno', alpha=0.8)
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('J')


plt.show()