#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import glob
import random
import cv2
from PIL import Image
import time

train_root = 'imgs/src_train/'
target_train_path = 'imgs/jpg_train'
test_root = 'imgs/src_test'
target_test_path = 'imgs/test'
new_train_path_root = 'imgs/train'

def read_img(path):
    img = cv2.imread(path)
    return img

def crop_img(img):
    x = 80
    y = 80

    return img[x:x+224, y:y+224, :]

def rotate_img(img, flag):
    rows , cols, ch = img.shape

    #if flag == 3:
    rotate_degree = random.uniform(-45, 45)
    #else :
    #    rotate_degree = random.uniform(0, 45)
    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), rotate_degree, 1)
    rotate_rd_img_1 = cv2.warpAffine(img, M, (cols, rows))
    #M = cv2.getRotationMatrix2D((rows / 2, cols / 2), rotate_degree_po, 1)
    #rotate_rd_img_2 = cv2.warpAffine(img, M, (cols, rows))

    return rotate_rd_img_1

def shift_img(img):
    rows, cols, ch = img.shape

    tx = random.uniform(-50, 50)
    ty = random.uniform(-50, 50)
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    shift_img = cv2.warpAffine(img, M, (cols, rows))

    return shift_img

def salt_and_pepper(src, percentage):
    noise_img = src;
    noise_num = int (percentage * src.shape[0] * src.shape[1])
    for i in range(noise_num):
        randX = int (np.random.uniform(0, noise_img.shape[1]))
        randY = int (np.random.uniform(0, noise_img.shape[0]))

        noise_img[randX, randY, 0] = 25
        noise_img[randX, randY, 1] = 20
        noise_img[randX, randY, 2] = 20

    return noise_img


def read_train_data():
    train_data = []
    label = []
    print ('read train file list')

    for i in range(12) :
        path = os.path.join(train_root, str(i))
        #target_path = os.path.join(target_train_path, str(i))
        new_train_path = os.path.join(new_train_path_root, str(i))

        #if not os.path.isdir(target_path):
        #    os.mkdir(target_path)

        if not os.path.isdir(new_train_path):
            os.mkdir(new_train_path)
        #files = glob.glob(path)

        print ('train class {}'.format(str(i)))
        for fl in os.listdir(path):
            temp = fl.split('.')
            new_name = temp[0] + '.jpg'
            #Image.open(path + '/' + fl).convert('RGB').save(
            #        target_path + '/' + new_name)

            #src_img = read_img(target_path + '/' + new_name)
            src_img = read_img(path + '/' + fl)
            if src_img == None :
                print fl
                continue
            #bigger_img = cv2.resize(img, (384, 384))
            img = cv2.resize(src_img, (224, 224))

            #flip_img = cv2.flip(img, 1)
            #crop_img = crop_img(bigger_img)
            #rotate_img = rotate_img(img)
            #percentage = np.random.uniform(0.0, 0.4)
            #noise_img = salt_and_pepper(img, percentage)

            new_224_name = new_train_path + '/' + temp[0]
            cv2.imwrite(new_224_name + '.jpg', img)
            #cv2.imwrite(new_224_name + 'cro.jpg', crop_img)
            #cv2.imwrite(new_224_name + 'hf.jpg', flip_img)
            #cv2.imwrite(new_224_name + 'rot.jpg', rotate_img)
            #cv2.imwrite(new_224_name + 'noi.jpg', noise_img)
            #cv2.imwrite(new_224_name + 'vf.jpg', v_flip_img)

            train_data.append(new_224_name + '.jpg')
            #train_data.append(new_224_name + 'cro.jpg')
            #train_data.append(new_224_name + 'hf.jpg')
            #train_data.append(new_224_name + 'rot.jpg')
            #train_data.append(new_224_name + 'noi.jpg')
            #train_data.append(new_224_name + 'vf.jpg')

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
