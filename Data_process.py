import os
import json
import numpy as np




# data: [..., [[feature], [label]], ...]
class DataLoader:

    def __init__(self, data, batch_size):
        self.data = np.array(data)
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
        
        return features, labels

        
    def get_data_size(self):

        return len(self.data)
    
    def get_data_dimention(self):

        feature_length = len(self.data[0][0])
        label_length = len(self.data[0][1])

        return feature_length, label_length
    
def read_data(data_path='./dataset_feature', train_ratio = 0.8):
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

train_, _ = read_data()

trainset = DataLoader(train_, batch_size=10)

for feature, label in trainset:
    print(len(feature[0]))
    print(len(label))
    # print(label)
    break