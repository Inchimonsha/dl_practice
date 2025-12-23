import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils


if __name__ == "__main__":
    # делаем датасет на обучающую и тестовую выборку
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()