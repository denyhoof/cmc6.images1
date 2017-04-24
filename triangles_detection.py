import cv2
import os
import numpy as np
from math import sin, cos, sqrt
import matplotlib.pyplot as plt

class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    def __truediv__(self, value):
        return Point2D(self.x / value, self.y / value)
    def __mul__(self, value):
        return Point2D(self.x * value, self.y * value)
    def norm(self):
        return sqrt(self.x * self.x + self.y * self.y)

def showHist(hist):
    plt.clf()
    plt.hist(hist, 15)
    plt.show()

def showImage(image):
    plt.clf()
    plt.imshow(image)
    plt.show()

def getIntersection(p1, p2):
    kEps = 0.001
    rho1, theta1 = p1
    rho2, theta2 = p2
    if abs(theta1 - theta2) < kEps:
        return None
    x = (rho2 * sin(theta1) - rho1 * sin(theta2)) / sin(theta1 - theta2)
    y = -(rho2 * cos(theta1) - rho1 * cos(theta2)) / sin(theta1 - theta2)
    return Point2D(x, y)

def drawLines(image, lines):
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

def VM(vector1, vector2):
    return vector1.x * vector2.y - vector2.x * vector1.y

def pointInTriangle(point, triangle):
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
    if pluscnt + nullcnt == 3 or minuscnt + nullcnt == 3:
        return True
    else:
        return False

def confidenceLine(image, a, b):
    tries = 50
    res = 0
    dy = b.y - a.y
    dx = b.x - a.x
    for i in range(tries):
        if image[int(a.y + dy * (i / tries)), int(a.x + dx * (i / tries))]:
            res += 1
    return res / tries

def checkConfidenceTriangle(image, triangle):
    kernel = np.ones((3,3),np.uint8)
    image_dil = cv2.dilate(image ,kernel, iterations = 1)
    res = confidenceLine(image_dil, triangle[0], triangle[1])
    res = min(res, confidenceLine(image_dil, triangle[0], triangle[2]))
    res = min(res, confidenceLine(image_dil, triangle[1], triangle[2]))
    #print(res / 3)
    return res

def triangleMidColor(block, triangle):
    h, w = block.shape[0], block.shape[1]
    color_sum = 0
    color_value = np.array([0, 0, 0])
    for y in range(h):
        for x in range(w):
            if pointInTriangle(Point2D(x, y), triangle):
                #print("Here")
                color_sum += 1
                color_value += block[y, x]
    if color_sum < 1:
        return [10000, 10000, 10000]
    print(color_value, color_sum)
    return color_value / max(1, color_sum)

def checkBlock(edges):
    max_line_size = 150
    min_line_size = 70

    lines = cv2.HoughLines(edges,1,np.pi/90,50)
    if lines is None:
        return None

    kDelta = 20
    for p1 in range(len(lines) - 2):
        for p2 in range(p1 + 1, len(lines - 1)):
            for p3 in range(p2 + 1, len(lines)):
                mr1 = getIntersection(lines[p1][0], lines[p2][0])
                mr2 = getIntersection(lines[p3][0], lines[p2][0])
                mr3 = getIntersection(lines[p1][0], lines[p3][0])
                if mr1 is not None and mr2 is not None and mr3 is not None:
                    l1 = (mr1 - mr2).norm()
                    l2 = (mr3 - mr2).norm()
                    l3 = (mr1 - mr3).norm()
                    if l1 < max_line_size and l1 > min_line_size and \
                        abs(l1 - l2) < kDelta and abs(l3 - l2) < kDelta and abs(l1 - l3) < kDelta:
                        return [mr1, mr2, mr3]
    return None

def center(points):
    mid = Point2D(0, 0)
    for point in points:
        mid += point
    return mid / len(points)

def areaTriangle(triangle):
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    return abs(a.x * b.y - a.y * b.x) / 2

def getRect(triangle):
    miny = 2000
    minx = 2000
    maxx = 0
    maxy = 0
    for p in triangle:
        miny = min(miny, p.y)
        minx = min(minx, p.x)
        maxy = max(maxy, p.y)
        maxx = max(maxx, p.x)
    return [miny, minx, maxy, maxx]

def getPointsInfo(image):
    image_median = cv2.medianBlur(image, 7)
    delta_image = np.abs(image - image_median.astype(np.int16))
    return [image, image_median, delta_image, np.maximum(delta_image[:, :, 0], 
                                                         np.maximum(delta_image[:, :, 1], delta_image[:, :, 2]))]

def HistColorFeaturesTriangle(blocks, triangles):
    color_hist = []
    for block, triangle in zip(blocks, triangles):
        color_hist.append(triangleMidColor(block, triangle))
    return color_hist

def checkImage(image, points, local_triangles, local_blocks):
    debug_maps = [image]
    h, w = image.shape[0], image.shape[1]
    image_median = cv2.medianBlur(image, 5)
    image_h = cv2.cvtColor(image_median, cv2.COLOR_BGR2HSV)[:,:,2]
    edges = cv2.Canny(image_h, 30, 60, apertureSize = 3)
    kBlockSize = 150
    kStep = 10
    res = image.copy()
    #detect triangles
    triangles = []
    kSimilarityTresh = 40
    sum_area = 0
    for y in range(0, h - kBlockSize, kStep):
        for x in range(0, w - kBlockSize, kStep):
            block = edges[y : y + kBlockSize, x : x + kBlockSize]
            mr = checkBlock(block)
            if mr is not None:
                for i in range(len(mr)):
                    mr[i].x = int(mr[i].x + x)
                    mr[i].y = int(mr[i].y + y)
                for i in range(len(triangles)):
                    if (center(triangles[i]) - center(mr)).norm() < kSimilarityTresh:
                        if abs(areaTriangle(mr) - sum_area / len(triangles)) < \
                            abs(areaTriangle(triangles[i]) - sum_area / len(triangles)):
                            sum_area -= areaTriangle(triangles[i])
                            sum_area += areaTriangle(mr)
                            triangles[i] = mr
                        #if areaTriangle(mr) < areaTriangle(triangles[i]) and \
                        #    areaTriangle(mr) / areaTriangle(triangles[i]) > 0.5:
                        #    triangles[i] = mr
                        break
                else:
                    triangles.append(mr)
                    sum_area += areaTriangle(mr)
    #print(len(triangles))
    #Thresh all bad rectangles: Hyp !(H > 50)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    triangles_filtered = []
    for triangle in triangles:
        rect = getRect(triangle)
        local_xy = Point2D(rect[1], rect[0])
        local_triangle = [triangle[0] - local_xy, triangle[1] - local_xy, 
                          triangle[2] - local_xy]
        image_hsv_cropped = image_hsv[rect[0] : rect[2], rect[1] : rect[3]]
        color_feature = triangleMidColor(image_hsv_cropped, local_triangle)
        cnf = checkConfidenceTriangle(edges, triangle)
        #print("Triangle features:", color_feature, cnf)
        if color_feature[0] > 50 or cnf < 0.3:
            continue
        else:
            #print("Yes, this is triangle!")
            triangles_filtered.append(triangle)
            points.append(center(triangle))
            local_triangles.append(local_triangle)
            local_blocks.append(image[rect[0] : rect[2], rect[1] : rect[3]])

        #print(cnf)
        #if cnf > 0.7:
        #    triangles_filtered.append(triangle)  
        #    points.append(center(triangle)) 
        #    local_triangles.append(local_triangle)
        #    local_blocks.append(image[rect[0] : rect[2], rect[1] : rect[3]])         

    for triangle in triangles_filtered:
        a, b, c = triangle
        cv2.line(res,(a.x, a.y),(b.x,b.y),(0,0,255),2)
        cv2.line(res,(c.x, c.y),(b.x,b.y),(0,0,255),2)
        cv2.line(res,(a.x, a.y),(c.x,c.y),(0,0,255),2)
    
    kernel = np.ones((3,3),np.uint8)
    image_dil = cv2.dilate(edges ,kernel, iterations = 1)
    debug_maps.append(image_dil)
    debug_maps.append(res)
    debug_maps.append(edges)
    debug_maps.append(image_h)
    return debug_maps