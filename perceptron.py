#!/usr/bin/env python
# encoding=utf-8

import numpy as np

class  Perceptron(object):
    """docstring for  Perceptron"""
    def __init__(self, num, activation):
        self.W = np.zeros(shape=num, dtype=np.float32)
        self.b = np.zeros(shape=1, dtype=np.float32)
        self.activation = activation

    def cal(self, vec):
        return self.activation(np.sum(self.W * vec) + self.b)

    def train(self, train_vecs, train_label, iterator_num, rate=0.1):
        for i in range(iterator_num):
            for (data, label) in zip(np.array(train_vecs), train_label):
                print('#' * 20)
                print('data', data)
                print('lable', label)
                print('deta w:', rate * (label - self.cal(data)) * data)
                print('deta b: ', rate * (label - self.cal(data)))
                print('before W', self.W)
                print('before b', self.b)
                self.W += rate * (label - self.cal(data)) * data
                self.b += rate * (label - self.cal(data))
                print('after W', self.W)
                print('after b', self.b)
                print('#' * 20)

    def __str__(self):
        return str(self.W) + str(self.b)


def activation(x):
    return 1 if x > 0 else 0



input_vecs = [[1,1], [0,0], [1,0], [0,1]]
labels = [1, 0, 0, 0]

def train_and_perceptron():
    p = Perceptron(num=2, activation=activation)
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print('*' * 20) 
    print(and_perception.W)
    print(and_perception.b)
    # 测试
    print('1 and 1 = {}'.format(and_perception.cal([1, 1])))
    print('0 and 0 = {}'.format(and_perception.cal([0, 0])))
    print('1 and 0 = {}'.format(and_perception.cal([1, 0])))
    print('0 and 1 = {}'.format(and_perception.cal([0, 1])))