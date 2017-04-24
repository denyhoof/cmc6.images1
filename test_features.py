import cv2
import os
import numpy as np
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from start import *

def showHist2(hist1, hist2, names):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(hist1[0], hist1[1], hist1[2], c='g', marker='o')
    ax.scatter(hist2[0], hist2[1], hist2[2], c='r', marker='o')
    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_zlabel(names[2])
    plt.show()

def loadTriangle(path):
    f = open(path)
    p1 = [int(i) for i in f.read().split()]
    return [Point2D(p1[1], p1[0]), Point2D(p1[3], p1[2]), Point2D(p1[5], p1[4])]

if __name__ == '__main__':
    output_dir = "output"
    input_cropped_tp = "input_tp"
    input_cropped_fp = "input_fp"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    test_files = os.listdir(input_cropped_tp)
    files_cnt = len(test_files) // 2

    cropped_images_tp = []
    local_triangles_tp = []
    for image_id in range(1, files_cnt + 1):
        file = str(image_id) + '.bmp'
        triangle = loadTriangle(input_cropped_tp + '/' + str(image_id) + '.txt')
        image = cv2.imread(input_cropped_tp + '/' + str(image_id) + '.bmp')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #cv2.imwrite(input_cropped_tp + '/' + str(image_id) + '.jpg', image)
        cropped_images_tp.append(image)
        local_triangles_tp.append(triangle)  
    hist_color = HistColorFeaturesTriangle(cropped_images_tp, local_triangles_tp)
    #hist_confidence = histConfidence(cropped_images_tp, local_triangles_tp)

    hist_red_tp = []
    hist_green_tp = []
    hist_blue_tp = []
    for elem in hist_color:
        hist_red_tp.append(elem[2])
        hist_green_tp.append(elem[1])
        hist_blue_tp.append(elem[0])

    test_files = os.listdir(input_cropped_fp)
    files_cnt = len(test_files) // 2

    cropped_images_fp = []
    local_triangles_fp = []
    for image_id in range(1, files_cnt + 1):
        file = str(image_id) + '.bmp'
        triangle = loadTriangle(input_cropped_fp + '/' + str(image_id) + '.txt')
        image = cv2.imread(input_cropped_fp + '/' + str(image_id) + '.bmp')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #cv2.imwrite(input_cropped_fp + '/' + str(image_id) + '.jpg', image)
        cropped_images_fp.append(image)
        local_triangles_fp.append(triangle)  
    hist_color = HistColorFeaturesTriangle(cropped_images_fp, local_triangles_fp)

    hist_red_fp = []
    hist_green_fp = []
    hist_blue_fp = []
    for elem in hist_color:
        hist_red_fp.append(elem[0])
        hist_green_fp.append(elem[1])
        hist_blue_fp.append(elem[2])
    showHist2([hist_red_tp, hist_green_tp, hist_blue_tp], [hist_red_fp, hist_green_fp, hist_blue_fp], ['V', 'S', 'H'])

