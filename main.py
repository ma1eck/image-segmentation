K = 200










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
image = ResizeWithAspectRatio(image, width=1280)

# to show image
# cv2.imshow('image',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




def distance(v1, v2): # v1 = (x, y, r, b, g)
    m  = 6
    pos1 = v1[:2]
    color1 = v1[2:]
    pos2 = v2[:2]
    color2 = v2[2:]
    pos_dist = np.linalg.norm(pos1 - pos2)
    color_dist = np.linalg.norm(color1 - color2)
    dist = (color_dist ** 2) * m + pos_dist ** 2
    dist = dist / (m+1)
    dist = dist ** 0.5

    return dist


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
print(k_means)