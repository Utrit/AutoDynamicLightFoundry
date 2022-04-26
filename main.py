import math
import random
import string
import cv2 as cv
import numpy as np
size = 512
jsonstart = '{"name": "data",\n"walls": [\n'
jsoninsert ='{"_id": "ID",\n"c": [ \nX1, \nY1, \nX2, \nY2 \n] \n}\n'
jsonend = ']}'
filename = "map.png"
sigma = 5
low_threshold = 36
high_threshold = 72
src = cv.imread(filename)
img = cv.resize(src.copy(), (size, size))

lut_in = [0, 127, 255]
lut_out = [0, 127, 255]

lut_8u = np.interp(np.arange(0, 256), lut_in, lut_out).astype(np.uint8)
#img = cv.LUT(img, lut_8u)

img_p = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(gray, (11, 11), 0)
gauss = cv.bilateralFilter(gauss,13,75,75)
wall_data = []


def auto_canny(image, sigma=0.50, low_thres=36, high_thres=72):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * (v - low_thres)))
    upper = int(min(255, (1.0 + sigma) * (v - high_thres)))
    edged = cv.Canny(image, lower, upper)
    return edged


def randomword(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def sqr_distance(pt1, pt2):
    dist = math.pow(pt1[0] - pt2[0], 2) + math.pow(pt1[1] - pt2[1], 2)
    return dist

def wall_similarity(wall1,wall2,threshold):
    sqr_threshold = math.pow(threshold,2)
    wall_len1 = sqr_distance(wall1[0],wall1[1])
    wall_len2 = sqr_distance(wall2[0], wall2[1])
    deltalen = abs(wall_len1-wall_len2)
    p1p3 = sqr_distance(wall1[0],wall2[0])
    p1p4 = sqr_distance(wall1[0],wall2[1])
    p2p3 = sqr_distance(wall1[1],wall2[0])
    p2p4 = sqr_distance(wall1[1],wall2[1])
    p1p1 = min(p1p3,p1p4)
    p2p2 = min(p2p3,p2p4)
    similarity = p1p1<sqr_threshold and p2p2<sqr_threshold and deltalen<300
    if(wall_len1>wall_len2):
        better_wall = 1
    else:
        better_wall = 2
    return similarity,better_wall

def merge_point_array(point, arr, tolerance):
    sqr_tolerance = math.pow(tolerance, 2)
    for p in arr:
        pt = (int(p[1]), int(p[0]))
        if sqr_distance(point, pt) > sqr_tolerance:
            return True
    return False


def on_low(data):
    global low_threshold
    low_threshold = data
    procces_image()


def on_high(data):
    global high_threshold
    high_threshold = data
    procces_image()


def on_sigma(data):
    global sigma
    sigma = data / 10
    procces_image()


def procces_image():
    global sigma, low_threshold, high_threshold, wall_output
    img_copy = img.copy()
    edges = auto_canny(gauss, sigma, low_threshold, high_threshold)
    canny_output = edges
    detect_polylines(canny_output,img_copy)
    #detect_polylines_hafa(canny_output,img_copy)
    #experemental_polylines(canny_output,img_copy)
    mix = np.concatenate((cv.cvtColor(edges, cv.COLOR_GRAY2BGR), img_copy), axis=1)
    cv.imshow("Preview", mix)

def detect_polylines_hafa(canny_output,img_copy):
    resize_coff = 1
    polylines = cv.HoughLinesP(cv.resize(canny_output,(size*resize_coff,size*resize_coff)),1,np.pi/80,10,np.array([]),5,5)
    for idx,lines in enumerate(polylines[:-1]):
        pt1 = (int(lines[0][0]/resize_coff), int(lines[0][1]/resize_coff))
        point2 = polylines[idx + 1]
        pt2 = (int(lines[0][2]/resize_coff), int(lines[0][3]/resize_coff))
        cv.line(img_copy, pt1, pt2, (0, 255, 0))

def reduce_wall(walls):
    for idx,wall in enumerate(walls):
        for idx2,wall_other in enumerate(walls[:idx]):
            sim,better_wall = wall_similarity(wall,wall_other,15)
            if sim:
                if better_wall==1:
                    walls.pop(idx2)
                else:
                    walls.pop(idx)
    return walls

def experemental_polylines(canny_output,img_copy):
    node_size = 8
    small_canny = cv.resize(canny_output.copy(),(size//2,size//2))
    rows, cols = small_canny.shape
    nodes = set([])
    for x in range(rows):
        for y in range(cols):
            if small_canny[y][x]>0:
                nodes.add((x//node_size,y//node_size))
    print(len(nodes))
    for x in nodes:
        pt = (x[0]*node_size*2,x[1]*node_size*2)
        cv.circle(img_copy,pt,8,(0,255,0))

def detect_polylines(canny_output,img_copy):
    global wall_data
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    wall_output = []
    for coord in contours:
        epsilon = 5
        coord = cv.approxPolyDP(coord, epsilon, False)
        for idx, point in enumerate(coord[:-1]):
            pt1 = (int(point[0][0]), int(point[0][1]))
            point2 = coord[idx + 1]
            pt2 = (int(point2[0][0]), int(point2[0][1]))
            #cv.line(img_copy, pt1, pt2, (0, 100+idx*10, 0))
            wall_output.append((pt1, pt2))
    if(len(wall_output)>2):
        wall_data = reduce_wall(wall_output.copy())
    else:
        wall_data = wall_output
    print(f'Reduced {len(wall_output)-len(wall_data)} walls of {len(wall_output)}')
    for walls in wall_data:
        cv.line(img_copy,walls[0],walls[1],(0,255,0))


def save_data(data):
    global wall_data
    height,width,_ = src.shape
    xcoff = width/size
    ycoff = height/size
    with open("Data.json", "w") as file:
        data_to_write = jsonstart
        for idx, coord in enumerate(wall_data):
            p1 = coord[0]
            p2 = coord[1]
            pattern = jsoninsert
            pattern = pattern.replace("ID", randomword(16))
            pattern = pattern.replace("X1", str(p1[0]*xcoff))
            pattern = pattern.replace("Y1", str(p1[1]*ycoff))
            pattern = pattern.replace("X2", str(p2[0]*xcoff))
            pattern = pattern.replace("Y2", str(p2[1]*ycoff))
            if (idx+1 < len(wall_data)):
                data_to_write += pattern + ','
            else:
                data_to_write += pattern
        data_to_write += jsonend
        file.write(data_to_write)


procces_image()

cv.createTrackbar("Low thres", "Preview", 36, 255, on_low)
cv.createTrackbar("High thres", "Preview", 72, 255, on_high)
cv.createTrackbar("Sigma", "Preview", 50, 100, on_sigma)
cv.createTrackbar("save", "Preview", 0, 1, save_data)
cv.waitKey(0)
cv.destroyAllWindows()
