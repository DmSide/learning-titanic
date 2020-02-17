from sklearn.cluster import KMeans
import pandas as pd
from skimage.io import imread
from skimage import img_as_float
import pylab
import math
import numpy as np

def PSNR(K, m, n):
    MAXI = 255.
    MSE = np.mean(K)
    return 20*math.log(MAXI, 10) - 10*math.log(MSE,10)

if __name__ == '__main__':
    image = imread('parrots.jpg')
    n = len(image)
    m = len(image[0])
    pylab.imshow(image)
    f_image = img_as_float(image)
    # TODO: Create matrix
    # f_matrix = [for x in f_imamge]
    cluster = KMeans(init='k-means++', random_state=241)
    cluster.fit(f_image, )
    psnr = PSNR(f_image, n, m)
    print(1)

