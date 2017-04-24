import cv2
import os
import numpy as np
from math import sin, cos, sqrt
import matplotlib.pyplot as plt
from start import *
from test_features import *

class PointTrimino:
    def __init__(self, x, y, value, color):
        self.x = x
        self.y = y
        self.value = value
        self.color = color

def isEmpty(points):
    for point in points:
        if point.color == 'yellow' or point.color == 'blue' or point.color == 'white':
            return 5000
    return len(points) * 2

def metricToClass(points, color, value):
    colors_cnt = {'white': 0, 'yellow' : 0, 'red' : 0, 'blue' : 0, 'green' : 0}
    for point in points:
        colors_cnt[point.color] += 1
    res = 0
    fl_red = 0

    if colors_cnt['white'] > 0:
        fl_red = 1
        if color != 'white':
            res += 5000
    else:
        if color == 'white':
            res += 5000

    if colors_cnt['yellow'] > 0:
        fl_red = 1
        if color != 'yellow':
            res += 5000
    else:
        if color == 'yellow':
            res += 5000

    if colors_cnt['blue'] > 0:
        fl_red = 1
        if color != 'blue':
            res += 5000
    else:
        if color == 'blue':
            res += 5000

    if colors_cnt['red'] > 0:
        if color != 'red':
            res += 3000
    else:
        if color == 'red':
            res += 3000
    #print(color, value)
    #print(colors_cnt)
    fl = 0
    mod = value
    if colors_cnt[color] >= value:
        colors_cnt[color] -= value
        fl = 1
        mod = 0
    else:
        mod -= colors_cnt[color]
        colors_cnt[color] = 0
    for pcolor in colors_cnt.keys():
        if pcolor == color:
            if not fl:
                res += colors_cnt[pcolor]
        else:
            if mod > 0:
                if colors_cnt[pcolor] >= mod:
                    res += mod + (colors_cnt[pcolor] - mod) * 2
                    mod = 0
                else:
                    res += colors_cnt[pcolor]
                    mod -= colors_cnt[pcolor]
            else:
                res += colors_cnt[pcolor] * 2
    res += 2 * mod
    #print(res)
    return res

def getPointsClass(points):
    best = 0
    best_val = isEmpty(points)
    classes = [['white', 1], ['green', 2], ['yellow', 3], ['blue', 4], ['red', 5]]
    for color, value in classes:
        tmp_val = metricToClass(points, color, value) 
        if tmp_val < best_val:
            best_val = tmp_val
            best = value
    return best

def getTriminoClass(triangle, points):
    points_corners = [[], [], []]
    for point in points:
        best = 0
        val = 1000000
        for i, corner in enumerate(triangle):
            tmp_val = (Point2D(point.x, point.y) - corner).norm()
            if tmp_val < val:
                val = tmp_val
                best = i
        if val > 10:
            points_corners[best].append(point)
    #print(points_corners)
    res = []
    for points_corner in points_corners:
        res.append(getPointsClass(points_corner))
    return res

def pointInsideTriangle(point, triangle):
    minuscnt = 0
    pluscnt = 0
    nullcnt = 0
    for i in range(3):
        vector1 = triangle[(i + 1) % 3] - triangle[i]
        vector2 = point - triangle[i]
        sgn = VM(vector1, vector2) 
        if sgn > 0:
            pluscnt += 1
        if sgn < 0:
            minuscnt += 1
        if sgn == 0:
            nullcnt += 1
    if pluscnt == 3 or minuscnt == 3:
        return True
    else:
        return False

def normalize(image, a, b):
    preva = np.min(image)
    prevb = np.max(image)
    if preva == prevb:
        return np.maximum(image, a)
    return (image - preva) / (prevb - preva) * (b - a)

def fillTriangleBlock(image, triangle, color):
    h, w = image.shape[0], image.shape[1]
    for y in range(h):
        for x in range(w):
            if not pointInsideTriangle(Point2D(x, y), triangle):
                image[y, x] = color
    return image

def extractCirclesMap(image, color, kPointThresh = 0.75, thresh=128):
    a, points_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    res = cv2.boxFilter(points_binary / 255, -1, (5,5))
    h, w = image.shape[0], image.shape[1]
    points = []
    kPointsEps = 5
    for y in range(h):
        for x in range(w):
            #if res[y, x] != 0:
            #    print(res[y, x])
            #print(y, x)
            if res[y, x] > kPointThresh :
                for it in range(len(points)):
                    if (Point2D(x, y) - Point2D(points[it].x, points[it].y)).norm() < kPointsEps:
                        if res[y, x] > points[it].value:
                            points[it] = PointTrimino(x, y, res[y, x], color)
                        break;
                else:
                    points.append(PointTrimino(x, y, res[y, x], color))
    #print(len(points))
    #for node in points:
    #    print(node.x, node.y, node.value, node.color)
    return points

def drawPoints(image, points):
    for point in points:
        cv2.circle(image, (point.x, point.y), radius=2, color = (0, 0, 0))
    return image

def testImage(image, triangle):
    fillTriangleBlock(image, triangle, triangleMidColor(image, triangle))
    maps = []
    
    white_points = np.minimum(image[:, :, 0], np.minimum(image[:, :, 1], image[:, :, 2]))
    points_white = extractCirclesMap(white_points, 'white', 0.5, 100)
    
    yellow_points = (image[:, :, 2].astype(np.int16) + image[:, :, 1].astype(np.int16) - 2 * image[:, :, 0].astype(np.int16)) / 2
    points_yellow = extractCirclesMap(np.maximum(0, np.minimum(255, yellow_points)).astype(np.uint8), 'yellow', 0.6, 70)

    blue_points = image[:, :, 0].astype(np.int16) - np.maximum(image[:, :, 1], image[:, :, 2])
    points_blue = extractCirclesMap(np.maximum(0, np.minimum(255, blue_points)).astype(np.uint8), 'blue', 0.6, 15)

    points_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0], image.shape[1]
    #green_points = np.zeros((h, w))
    #plt.clf()
    #plt.imshow(green_points_hsv[:,:, 0])
    #plt.colorbar()
    #plt.show()
    #for y in range(h):
    #    for x in range(w):
    #        if points_hsv[y, x, 0] > 10 and points_hsv[y, x, 0] < 100 and points_hsv[y, x, 1] < 150:# and points_hsv[y, x, 2] < 80:
    #            green_points[y, x] = 255
    #        else:
    #           green_points[y, x] = 0
    green_points = 255 - np.maximum(image[:,:,0], np.maximum(image[:,:,1], image[:,:,2])).astype(np.int16)   
    #points_green = extractCirclesMap(green_points.astype(np.uint8), 'green', 0.4, 50)    
    points_green = extractCirclesMap(green_points.astype(np.uint8), 'green', 0.5, 180)

    red_points = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if (points_hsv[y, x, 0] < 5 or points_hsv[y, x, 0] > 170) and points_hsv[y, x, 1] > 160 and points_hsv[y, x, 2] > 80:
                red_points[y, x] = 255
            else:
                red_points[y, x] = 0
    points_red = extractCirclesMap(red_points.astype(np.uint8), 'red', 0.4, 50)

    points = []
    for point in points_white:
        points.append(point)
    for point in points_yellow:
        points.append(point)
    for point in points_red:
        points.append(point)
    for point in points_blue:
        points.append(point)
    for point in points_green:
        points.append(point)

    res = getTriminoClass(triangle, points)
    #print('RES', res)

    #image = drawPoints(image, points_white)
    #image = drawPoints(image, points_yellow)
    #image = drawPoints(image, points_red)
    #image = drawPoints(image, points_blue)
    #image = drawPoints(image, points_green)
    
    #maps.append(image)
  
    #maps.append(white_points)
    #maps.append(yellow_points)
    #maps.append(red_points)
    #maps.append(blue_points)
    #maps.append(green_points)
   
    return [res, maps]