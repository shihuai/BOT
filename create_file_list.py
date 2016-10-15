#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import glob
import random
from PIL import Image

train_root = 'imgs/src_train'
target_train_path = 'imgs/train'
test_root = 'imgs/src_test'
target_test_path = 'imgs/test7'

def read_train_data():
    train_data = []
    label = []
    print ('read train file list')

    for i in range(12) :
        path = os.path.join(train_root, str(i))
        target_path = os.path.join(target_train_path, str(i))

        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        #files = glob.glob(path)

        print ('train class {}'.format(str(i)))
        for fl in os.listdir(path):
            print fl
            temp = fl.split('.')
            new_name = temp[0] + '.jpg'
            Image.open(path + '/' + fl).convert('RGB').save(
                                target_path + '/' + new_name)

            train_data.append(target_path + '/' + new_name)
            label.append(i)

    print ('finished search flods')
    return train_data, label

def read_train_data_and_change_file_format():

    file_list, label = read_train_data()

    size = len(file_list)
    index = np.arange(size)
    random.shuffle(index)
    test_size = int(size * 0.1)

    print ('start write file list')
    fi = open('val.txt', 'w')

    for i in range(test_size):
        fi.write(file_list[index[i]])
        fi.write(' ')
        fi.write(str(label[index[i]]))
        fi.write('\n')
    fi.close()

    fi = open('train.txt', 'w')
    for i in range(test_size, size):
        fi.write(file_list[index[i]])
        fi.write(' ')
        fi.write(str(label[index[i]]))
        fi.write('\n')
    fi.close()

    print ('finished write!')

def read_test_data():
    test_data = []

    print 'start read test data'

    if not os.path.isdir(target_test_path):
        os.mkdir(target_test_path)

    for fl in os.listdir(test_root):
        temp = fl.split('.')
        new_fl = target_test_path+ '/' + temp[0] + '.jpg'
        Image.open(test_root + '/' + fl).convert('RGB').save(new_fl)

        test_data.append(new_fl)

    print ('finished read test data')
    return test_data

def read_test_data_and_change_file_format():
    test_data = read_test_data()

    fi = open('test6.txt', 'w')

    for name in test_data:
        fi.write(name + '\n')
    fi.close()

#read_train_data_and_change_file_format()
read_test_data_and_change_file_format()
