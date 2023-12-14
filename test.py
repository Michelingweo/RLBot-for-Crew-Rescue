import numpy as np

x = np.array([[3.13, 4.07, 1.18, 6.8],
            [3.93, 3.37170, 3.82338, 3.1523],
            [4.74817409e-1, 6.38779, 5.985331, 9.040],
            [2.6863, 3.51817919e-1, 3.197700, 1.79883971e-02]
               ])
# x= np.array([-20.889393859003302,-20.763846570528223,-17.836916216831188, -24.841725545695887])

# c = np.max(x)
# b = x-c
# print(b.dtype)
# print(np.exp(x-c))


def softmax(x):
    # Subtract the max for numerical stability
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / sum_exp_x


# def softmax(x):
#     # Subtract the max for numerical stability
#     c = np.max(x, axis=0, keepdims=True)
#     exp_x = np.exp(x - c)
    
#     # Sum along the rows
#     sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)
    
#     return exp_x / sum_exp_x

# def softmax(x):
 
#     c = np.max(x)
    
#     exp_a = np.exp(x-c)
#     sum_exp_a = np.sum(exp_a)
#     return exp_a/np.sum(sum_exp_a)

x 