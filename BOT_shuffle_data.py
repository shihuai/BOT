#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import glob
import random
import cv2
from PIL import Image

root_path = 'imgs/train'

def read_train_data():
    train_data = []
    label = []
    print ('read train file list')

    for i in range(12) :
        path = os.path.join(root_path, str(i))
        #target_path = os.path.join(target_train_path, str(i))

        #if not os.path.isdir(target_path):
        #    os.mkdir(target_path)

        print ('train class {}'.format(str(i)))
        for fl in os.listdir(path):
            #img = read_img(path + '/' + fl)
            #if img == None :
            #    continue
            #bigger_img = cv2.resize(img, (384, 384))
            #img = cv2.resize(img, (224, 224))
            #flip_img = cv2.flip(img, 1)
            #crop = crop_img(bigger_img)
            #rotate_ne, rotate_po = rotate_img(img)
            #shift = shift_img(img)

            #new_224_name = new_train_path + '/' + temp[0]
            #cv2.imwrite(new_224_name + '.jpg', img)
            #cv2.imwrite(new_224_name + 'c.jpg', crop)
            #cv2.imwrite(new_224_name + 'f.jpg', flip_img)
            #cv2.imwrite(new_224_name + 'rn.jpg', rotate_ne)
            #cv2.imwrite(new_224_name + 'rp.jpg', rotate_po)
            #cv2.imwrite(new_224_name + 's.jpg', shift)

            train_data.append(path + '/' + fl)
            #train_data.append(new_224_name + 'c.jpg')
            #train_data.append(new_224_name + 'f.jpg')
            #train_data.append(new_224_name + 'rn.jpg')
            #train_data.append(new_224_name + 'rp.jpg')
            #train_data.append(new_224_name + 's.jpg')

            label.append(i)
            #label.append(i)
            #label.append(i)
            #label.append(i)
            #label.append(i)
            #label.append(i)

    print ('finished search flods')
    return train_data, label

def read_train_data_and_change_file_format():

    file_list, label = read_train_data()

    size = len(file_list)
    index = np.arange(size)
    random.shuffle(index)
    test_size = int(size * 0.2)

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

read_train_data_and_change_file_format()
