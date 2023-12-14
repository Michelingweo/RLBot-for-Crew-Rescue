import numpy as np
from NN.NNfunctions import *
from NN.deep_convnet import *
from NN.optimizer import *
import pickle
import torch
from model1 import Trainer
import matplotlib.pyplot as plt
from Data_process import *


class Model2:
    def __init__(self, input_dim=(3, 50, 50),
        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},\
                   hidden_size=40, output_size=2, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size=input_dim[1]
        conv_output_size=(input_size - filter_size + 2*filter_pad)/filter_stride + 1
        pool_output_size=int(filter_num*(conv_output_size/2)**2)

    #     初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std* np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std* np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std* np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)


        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return softmax(x)


    def loss(self, x, t):
        for layer in self.layers.values():
            x = layer.forward(x)
        loss = self.last_layer.forward(x, t)
        return loss

    def accuracy(self, test_set, batch_size=100):
        
        # if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(test_set.setLen / batch_size)):
            x_test, y_test = test_set.__next__()
            
            y_pred = self.predict(x_test)
            # print('raw_pred:', y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            # print('label_pred:', y_pred)
            y_test = np.argmax(y_test, axis=1)
            acc += np.sum(y_pred == y_test)

        return acc / test_set.setLen
    
    
    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        #设定
        grads={}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

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
        
        for i, layer_name in enumerate(self.layers.keys()):
            self.layers[layer_name].W = self.params['W' + str(i+1)]
            self.layers[layer_name].b = self.params['b' + str(i+1)]
    
    
if __name__ == '__main__':

    x_bot, y_bot, x_crew, y_crew = 1,2,3,4
    k = 2

    #Para
    max_epochs = 100
    lr = 0.00001
    bs = 64

    # data & model init
    print('Start loading dataset.')
    
    if os.path.exists('dataset_matrix/matrix_trainset.npy'):
        trainset_ = np.load('dataset_matrix/matrix_trainset.npy', allow_pickle=True)
        testset_ = np.load('dataset_matrix/matrix_testset.npy', allow_pickle=True) 
    else:
        trainset_, testset_ = Model2_read_data_matrix()
        
        np.save('dataset_matrix/matrix_trainset.npy',trainset_)
        np.save('dataset_matrix/matrix_testset.npy',testset_)
    
    train_set = DataLoader(trainset_, batch_size=bs)
    test_set = DataLoader(testset_, batch_size=bs)
    
    input_size, label_size = train_set.get_data_dimention()
    
    print('Dataset loaded.')
    
    # x_train, y_train, x_test, y_test 
    network = Model2()
    
    
    trainer = Trainer(network, train_set, test_set,
                      epochs=max_epochs, mini_batch_size=bs,
                      optimizer='Adam', optimizer_param={'lr': lr},
                      evaluate_sample_num_per_epoch=100)
    trainer.train()

    # 保存参数
    network.save_params("model2_params.pkl")
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
    plt.savefig('model2_acc.png')

