#!/usr/bin/env python
# encoding=utf-8

import numpy as np

class  Linear_unit(object):
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
                detal_W = rate * (label - self.cal(data)) * data
                detal_b = rate * (label - self.cal(data))
                self.W += detal_W
                self.b += detal_b
                print('after W', self.W)
                print('after b', self.b)
                print('#' * 20)

    def __str__(self):
        return str(self.W) + str(self.b)


def activation(x):
    return x



input_vecs = [[5], [3], [8], [1.4], [10.1]]
labels = [5500, 2300, 7600, 1800, 11400]

def train_linear_unit():
    p = Linear_unit(num=1, activation=activation)
    p.train(input_vecs, labels, 10, 0.01)
    return p

if __name__ == '__main__':
    # 训练and感知器
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print('*' * 20) 
    print(linear_unit.W)
    print(linear_unit.b)
    # 测试
    print('3.4: {}'.format(linear_unit.cal([3.4])))
    print('15: {}'.format(linear_unit.cal([15])))
    print('1.5 {}'.format(linear_unit.cal([1.5])))
    print('6.3: {}'.format(linear_unit.cal([6.3])))