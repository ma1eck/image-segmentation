K = 50










import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
import matplotlib.image as mpimg
from PIL import Image





def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


image = cv2.imread('R.jpeg')
image = ResizeWithAspectRatio(image, width=380)

# to show image
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




def calculate_distance(v1, v2): # v1 = (x, y, r, b, g)
    m  = 6
    pos1 = v1[:2]
    color1 = v1[2:]
    pos2 = v2[:2]
    color2 = v2[2:]
    pos_dist = np.linalg.norm(pos1 - pos2)
    color_dist = np.linalg.norm(color1 - color2)
    distance = (color_dist ** 2) * m + pos_dist ** 2
    distance = distance / (m+1)
    distance = distance ** 0.5

    return distance


height, width, channels = image.shape
nodes = np.empty((height * width, 5), dtype= np.uint32)

counter = 0
for y in range(height):
    for x in range(width):
        b, g, r = image[y][x]
        nodes[counter] = [x, y, r, g, b]
        counter += 1


k_means = np.empty([K, 5], dtype= np.uint32 )
for i in range(K):
    k_means[i] =  np.random.randint(low = [0, 0, 0, 0, 0], high= [width, height, 256, 256 ,256], dtype=np.uint32)

def initial_center_of_nodes_and_nodes_of_centers():
    global center_of_nodes
    global nodes_of_centers
    center_of_nodes = []
    nodes_of_centers = []
    for i in range(K):
        nodes_of_centers.append([])

def allocate_centers():
    for i, node in enumerate(nodes):
        min_dist = float('inf')
        min_index = None
        for j, center in enumerate(k_means):
            distance = calculate_distance(node, center)
            if distance < min_dist:
                min_index = j
                min_dist = distance
        center_of_nodes.append(min_index)
        nodes_of_centers[min_index].append(node)

initial_center_of_nodes_and_nodes_of_centers()
allocate_centers()

def update_k_means():
    global k_means, nodes_of_centers
    for i in range(K):
        if len(nodes_of_centers[i]) != 0:
            mean_of_nodes = np.mean(nodes_of_centers[i], axis = 0)
            k_means[i] = mean_of_nodes
for i in range(5):
    update_k_means()
    initial_center_of_nodes_and_nodes_of_centers()
    allocate_centers()


    






new_image = np.zeros((height,width,3), np.uint8)

for i, node in enumerate(nodes):
    nearest_center = center_of_nodes[i]
    rgb =  k_means[ nearest_center] [2:]
    x = node[0]
    y = node[1]
    new_image[y][x] = rgb

cv2.imshow('image',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()