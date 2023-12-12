import numpy as np

# x = np.array([0.2, 0.87, 0.99, 23.1, 13.25])
x= np.array([-20.889393859003302,-20.763846570528223,-17.836916216831188, -24.841725545695887])

c = np.max(x)
b = x-c
print(b.dtype)
print(np.exp(x-c))


def softmax(x):
    print("x:", x)  # 打印 x 查看其内容
    print("x type:", type(x))  # 打印 x 的类型
    c = np.max(x)
    
    print("x-c:", x-c)  # 打印 x 查看其内容
    print("x-c type:", type(x-c))  # 打印 x 的类型
    
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a/np.sum(sum_exp_a)

print(softmax(x))