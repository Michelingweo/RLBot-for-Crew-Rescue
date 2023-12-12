import numpy as np
from NN.NNfunctions import *
from NN.deep_convnet import *
from NN.optimizer import *
import pickle
import torch

import matplotlib.pyplot as plt
from Data_process import DataLoader, read_data
# input:


# ship layout [0, 1, 0, 1]
# bot pos: (x,y)
# argmax crew prob pos: (x,y) 
# alien detection area: (2k+1)^2 ---each pos is the alien probability


class SpaceShipMLP:
    def __init__(self, input_size, output_size):
        pre_node_nums = np.array([input_size])
        wight_init_scale = np.sqrt(2.0 / pre_node_nums)  # recommended init value when using ReLu
        
        self.params = {}
        self.params['W0'] = wight_init_scale * np.random.randn(input_size, 2*input_size)
        self.params['b0'] = np.zeros(2*input_size)
        self.params['W3'] = wight_init_scale * np.random.randn(2*input_size, int(input_size/2))
        self.params['b3'] = np.zeros(int(input_size/2))
        self.params['W5'] = wight_init_scale * np.random.randn(int(input_size/2), output_size)
        self.params['b5'] = np.zeros(output_size)
        self.layers = []
 
        self.layers.append(Affine(self.params['W0'], self.params['b0']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W3'], self.params['b3']))
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W5'], self.params['b5']))

        self.last_layer = SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, test_set, batch_size=100):
        
        # if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(test_set.setLen / batch_size)):
            x_test, y_test = test_set.__next__()
            
            y_pred = self.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            acc += np.sum(y_pred == y_test)

        return acc / x_test.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 3, 5)):
            grads['W' + str(layer_idx)] = self.layers[layer_idx].dW
            grads['b' + str(layer_idx)] = self.layers[layer_idx].db

        # print(grads.keys())
        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]

#training
class Trainer:

    def __init__(self, network, train_set, test_set,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr': 0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.train_set = train_set
        self.test_set = test_set
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprpo': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = train_set.setLen
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        
        x_batch, t_batch = self.train_set.__next__()

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            
            x_train_sample, t_train_sample = self.train_set.__next__()
            x_test_sample, t_test_sample = self.test_set.__next__()
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.train_set.__next__(bs=t)
                x_test_sample, t_test_sample = self.test_set.__next__(bs=t)

            train_acc = self.network.accuracy(self.test_set)
            test_acc = self.network.accuracy(self.test_set)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(
                    test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))


if __name__ == '__main__':

    x_bot, y_bot, x_crew, y_crew = 1,2,3,4
    k = 2

    ship_layout = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    bot_pos = torch.tensor([x_bot, y_bot], dtype=torch.float32)
    crew_prob_pos = torch.tensor([x_crew, y_crew], dtype=torch.float32)
    alien_area = torch.randn((2*k+1)**2)  # Example alien area data

    #Para
    max_epochs = 100
    lr = 0.001
    bs = 64

    # data & model init
    trainset_, testset_ = read_data()
    
    train_set = DataLoader(trainset_, batch_size=bs)
    test_set = DataLoader(testset_, batch_size=bs)
    
    input_size, label_size = train_set.get_data_dimention()
    
    # x_train, y_train, x_test, y_test 
    network = SpaceShipMLP(input_size=input_size, output_size=label_size)

    
    trainer = Trainer(network, train_set, test_set,
                      epochs=max_epochs, mini_batch_size=bs,
                      optimizer='Adam', optimizer_param={'lr': lr},
                      evaluate_sample_num_per_epoch=100)
    trainer.train()

    # 保存参数
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    # 绘制图形
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()