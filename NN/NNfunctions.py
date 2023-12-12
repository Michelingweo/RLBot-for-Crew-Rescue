import numpy as np


def indentity_funciton(x):
    return x


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1/(1+np.exp(x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def identity_function(x):
    return x


def Relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad


def mean_square_error(y, t):
    return 0.5*np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))


def softmax(x):
    # Subtract the max for numerical stability
    c = np.max(x, axis=1, keepdims=True)
    _ = x-c
    _ = _.astype(float)
    exp_x = np.exp(_)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x

# def softmax(x):

#     c = np.max(x)
#     _ = x-c
#     _ = _.astype(float)
#     exp_a = np.exp(_)
#     sum_exp_a = np.sum(exp_a)
#     return exp_a/np.sum(sum_exp_a)


def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)+f(-h)/(2*h))


def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)#produce an arrary with same size with x

    for idx in range(x.size):
        tmp_val = x[idx]
        #calculate f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)
        #calculate f(x-h)
        x[idx]=tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x=init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad

    return x


def softmax_loss(X,t):
    y=softmax(X)
    return cross_entropy_error(y,t)