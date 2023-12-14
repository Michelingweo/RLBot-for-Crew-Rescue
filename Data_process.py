import os
import json
import numpy as np


# data: [..., [[feature], [label]], ...]
class DataLoader:

    def __init__(self, data, batch_size):
        # print(data)
        self.data = np.array(data, dtype=object)
        # print(self.data.shape)
        self.batch_size = batch_size
        self.index = 0
        self.setLen = len(self.data)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self, bs=None):
        if bs == None:
            bs = self.batch_size
        
        batch_mask = np.random.choice(self.setLen, bs)
        # print(batch_mask)
        # print(batch_mask.dtype)
        batch_data = self.data[batch_mask]
        
        features = []
        labels = []
    
        for item in batch_data:
            # 假设每个item都是 [特征值, 标签值] 的形式
            if len(item) == 2:  
                features.append(item[0])  # 特征值
                labels.append(item[1])    # 标签值
            else:
                raise ValueError("Each item should have both feature and label")
        
        return np.array(features, dtype=object), np.array(labels, dtype=object)

        
    def get_data_size(self):

        return len(self.data)
    
    def get_data_dimention(self):

        feature_shape = len(self.data[0][0])
        label_shape = len(self.data[0][1])

        return feature_shape, label_shape
    

def Model1_read_dataset_feature(data_path='./dataset_feature', train_ratio = 0.8):
    # 指定包含 txt 文件的目录路径
    directory_path = data_path

    dataset = []

    # 获取目录下所有文件
    files = os.listdir(directory_path)

    # 循环遍历每个文件
    for file_name in files:

        file_dataset = []
        if file_name.endswith('.txt'):  # 确保是 txt 文件
            file_path = os.path.join(directory_path, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    row = eval(line)  # 从文本行中获取数据并转换为整数
                    file_dataset.append(row)
        
        dataset.extend(file_dataset[:-1])

    # 使用 NumPy 来 shuffle 列表
    np.random.shuffle(dataset)

    # 计算分割点
    split_index = int(len(dataset) * train_ratio)

    # 分割列表
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    
    return train_set, test_set


def Model2_read_data_feature(data_path='./dataset_feature', train_ratio = 0.8):
    # 指定包含 txt 文件的目录路径
    directory_path = data_path

    success_dataset = []
    fail_dataset = []

    total_dataset = []

    # 获取目录下所有文件
    files = os.listdir(directory_path)

    # 循环遍历每个文件
    for file_name in files:

        file_dataset = []
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines[:-1]:
                    row = eval(line)
                    # print(row[0])
                    file_dataset.append(row[0])
                
                success = eval(lines[-1])
                if success == 0:
                    success_onehot = [1, 0]
                elif success == 1:
                    success_onehot = [0, 1]
                else:
                    raise ValueError


        for i in range(len(file_dataset)):

            data_row = file_dataset[i]
            data_row = [data_row, success_onehot]
            file_dataset[i] = data_row
            # print(data_row)
        
        if success:
            success_dataset.extend(file_dataset)
        else:
            fail_dataset.extend(file_dataset)

    success_num = len(success_dataset)
    fail_num = len(fail_dataset)

    # print(success_num, fail_num)

    total_dataset.extend(success_dataset[:min(success_num, fail_num)])
    total_dataset.extend(fail_dataset[:min(success_num, fail_num)])

    # print(len(total_dataset))

    # 使用 NumPy 来 shuffle 列表
    np.random.shuffle(total_dataset)

    # 计算分割点
    split_index = int(len(total_dataset) * train_ratio)

    # 分割列表
    train_set = total_dataset[:split_index]
    test_set = total_dataset[split_index:]
    
    return train_set, test_set


def Model2_read_data_matrix(data_path='./dataset_matrix', train_ratio = 0.8):
    # 指定包含 txt 文件的目录路径
    directory_path = data_path

    success_dataset = []
    fail_dataset = []

    total_dataset = []

    # 获取目录下所有文件
    files = os.listdir(directory_path)
    
    # 循环遍历每个文件
    for file_name in files:
        
        file_dataset = []
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            print(file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines[:-1]:
                    row = eval(line)
                    # print(row[0])
                    file_dataset.append(row[0])
                
                success = eval(lines[-1])
                # print('success:', success)
                if success == 0:
                    success_onehot = [1, 0]
                    
                elif success == 1:
                    success_onehot = [0, 1]
                    
                else:
                    raise ValueError


        for i in range(len(file_dataset)):

            data_row = file_dataset[i]
            data_row = [data_row, success_onehot]
            file_dataset[i] = data_row
            # print(data_row)
        
        if success:
            success_dataset.extend(file_dataset)
        else:
            fail_dataset.extend(file_dataset)
        
        
    # total_dataset.extend(success_dataset)
    # total_dataset.extend(fail_dataset)
        
    success_num = len(success_dataset)
    fail_num = len(fail_dataset)

    print("succes to fail ratio:",success_num, fail_num)

    total_dataset.extend(success_dataset[:min(success_num, fail_num)])
    total_dataset.extend(fail_dataset[:min(success_num, fail_num)])

    print("dataset size:", len(total_dataset))

    # 使用 NumPy 来 shuffle 列表
    np.random.shuffle(total_dataset)

    # 计算分割点
    split_index = int(len(total_dataset) * train_ratio)

    # 分割列表
    train_set = total_dataset[:split_index]
    test_set = total_dataset[split_index:]
    
    return train_set, test_set



if __name__ == "__main__":

    train_, _ = Model2_read_data_matrix()

    # print(train_)

    trainset = DataLoader(train_, batch_size=10)

    print(trainset.get_data_dimention())

