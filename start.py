import os
from triangles_detection import *
from blocks_detection import *

def solve(image):
    local_triangles = []
    local_blocks = []
    points = []
    maps = checkImage(image, points, local_triangles, local_blocks)
    ans = []
    res_maps = []
    for m in maps:
        res_maps.append(m)
    for point, triangle, block in zip(points, local_triangles, local_blocks):
        print("Check triangle: ", point.x, point.y)
        points_info, local_maps = testImage(block.copy(), triangle)
        for m in local_maps:
            res_maps.append(m)
        mr = []
        mr.append(int(point.x))
        mr.append(int(point.y))
        for i in points_info:
            mr.append(i)
        ans.append(mr)
    return ans, res_maps

def saveAnswer(path, answer):
    f = open(path, 'w')
    f.write(str(len(answer)) + '\n')
    for node in answer:
        f.write(str(node[0]) + ', ' + str(node[1]) + '; ' + str(node[2]) + ', ' + \
                str(node[3]) + ', ' + str(node[4]) + '\n')
    f.close()

if __name__ == '__main__':
    output_dir = "output"
    input_dir = "input"
    test_files = os.listdir(input_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for file in test_files:
        print(file)
        #ile = '31.bmp'
        image = cv2.imread(input_dir + '/' + file)
        answer, res = solve(image)
        for i, img in enumerate(res):
            cv2.imwrite(output_dir + '/' + file + '_' + str(i) + '.bmp', img)
        saveAnswer(output_dir + '/' + file + '.txt', answer)
        #print(answer)
        #break