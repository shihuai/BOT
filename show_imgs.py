#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
import os

fi = open('result_googlenet_test_6.txt', 'r')
files = fi.readlines()
fi.close()

fi = open('name.txt', 'r')
classes = fi.readlines()
fi.close()

file_root = 'imgs/test6/'

for fi in files:
    temp = fi.split('\t')
    path = file_root + temp[0] + '.jpg'

    img = cv2.imread(path)

    img = cv2.resize(img, (448, 448))
    top1 = int(temp[1])
    top2 = int(temp[3])

    cls_1 = classes[top1]
    cls_2 = classes[top2]

    #print temp[1], temp[3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'top1: ' + cls_1, (0, 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, 'top2: ' + cls_2, (0, 45), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey(0)

