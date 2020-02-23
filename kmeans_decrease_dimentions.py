from sklearn.cluster import KMeans
import pandas as pd
from skimage.io import imread
from skimage import img_as_float
import pylab
import math
import numpy as np


def PSNR(K):
    MAXI = 1  # 255.
    MSE = np.mean(K)
    return 20*math.log(MAXI, 10) - 10*math.log(MSE, 10)


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


if __name__ == '__main__':
    N = 11
    image = imread('parrots.jpg')
    #  pylab.imshow(image)
    f_image = img_as_float(image)
    # TODO: Create matrix
    # f_matrix = [for x in f_imamge]
    w, h, d = original_shape = tuple(f_image.shape)
    assert d == 3
    image_array = np.reshape(f_image, (w * h, d))

    # r = np.array([image[:, :, 0].ravel()]).T
    # g = np.array([image[:, :, 1].ravel()]).T
    # b = np.array([image[:, :, 2].ravel()]).T
    # result = np.hstack((r, g))
    # result = np.hstack((result, b))
    cluster = KMeans(n_clusters=N, init='k-means++', random_state=241)
    cluster.fit(image_array)
    labels = cluster.predict(image_array)
    image_pred = recreate_image(cluster.cluster_centers_, labels, w, h)
    # psnr = PSNR(f_image)
    print(10 * np.log10(1.0 / np.mean((f_image - image_pred) ** 2)))
    # mse = np.mean((image1 - image2) ** 2)
    # psnr = 10 * math.log10(float(1) / mse)

    # image = imread('d:\parrots.jpg')
    # image2 = skimage.img_as_float(image)
    # w, h, d = original_shape = tuple(image2.shape)
    # assert d == 3
    # image_array = np.reshape(image2, (w * h, d))
    # kmeans = KMeans(n_clusters=N, init='k-means++', random_state=241)
    # kmeans.fit(image_array)
    # labels = kmeans.predict(image_array)
    #
    # image_pred = recreate_image(kmeans.cluster_centers_, labels, w, h)
    # print(10 * np.log10(1.0 / np.mean((image2 - image_pred) ** 2)))

