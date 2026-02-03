from math import e

# Activation function

def sigmoid(x):
    ''' Сигмоид '''
    return 1 / (1 + e**(-x))

def hyperbolic_tangent(x):
    ''' Гиперболический тангенс '''
    return (e**(2 * x) - 1) / (e**(2 * x) + 1)

# Error function

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

def Root_MSE(I, a, n):
    '''

    :param I:
    :param a:
    :param n:
    :return:
    '''

def Arctan(I, a, n):
    '''

    :param I:
    :param a:
    :param n:
    :return:
    '''

    return

# Train function

def fit_input(I, w, n):
    H_input = 0
    for i in range(n):
        H_input += I[i] * w[i]
    return H_input

def fit_output(f_actv, H_input):
    H_output = f_actv(H_input)
    return H_output

# Epoch function

def f_epoch():
    for i in range(max_epoch):
        return

# def fit():
#     for i in range():


# other
def derv(x):
    return (1 - x) * x

def delta_output(out_ideal, out_actual):
    der = derv(out_actual)
    return (out_ideal - out_actual) * der

def delta_input(H_output, w, delta, n):
    rsum = 0
    der = derv(H_output)
    for i in range(n):
        rsum += w[i] * delta[i]
    return der * rsum

def grad(H_output, delta):
    return H_output * delta

def delta_new_w(grad, delta_w_prev, E, A):
    return E * grad + A * delta_w_prev

if __name__ == "__main__":
    I = [1, 0]
    a = []
    w1 = [0.45, -0.12]
    w2 = [0.78, 0.13]
    w_out = [1.5, -2.3]

    max_epoch = 3
    train_set = 2

    h1input = fit_input(I, w1, train_set)
    h1output = fit_output(sigmoid, h1input)

    h2input = fit_input(I, w2, train_set)
    h2output = fit_output(sigmoid, h2input)

    I_out = [h1output, h2output]
    o1input = fit_input(I_out, w_out, train_set)
    o1output = fit_output(sigmoid, o1input)
    a.append(o1output)

    error = MSE(I, a, 1)
    print(o1output, error)

    deltas_out = []
    delta1_out = delta_output(I[0], o1output)
    deltas_out.append(delta1_out)

    delta1_input = delta_input(h1output, w_out, deltas_out, 1)

    E = 0.7 # скорость обучения
    A = 0.3 # момент

    delta_w_prev = 0
    grad_w5 = grad(h1output, delta1_out)
    delta_w5 = delta_new_w(grad_w5, delta_w_prev, E, A)
    w_out[0] = w_out[0] + delta_w5

    grad_w6 = grad(h2output, delta1_out)
    delta_w6 = delta_new_w(grad_w6, delta_w_prev, E, A)
    w_out[1] = w_out[1] + delta_w6

    print(w_out)