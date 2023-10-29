import numpy as np
import cv2
import os

from suppression import *
from threshold import thresholding

# Сохранять изображения или нет
save_images = 0

gaussian = (1 / 16) * np.array([[1, 2, 1], 
                                [2, 4, 2],
                                [1, 2, 1]])
sobelx = np.array([[-1, -2, -1], 
                    [0, 0, 0],
                    [1, 2, 1]])
sobely = np.transpose(sobelx)

# Считывание изображения
image = cv2.imread(os.getcwd() + '/images/scale_1200.jpeg', cv2.IMREAD_COLOR)

# Преобразование изображения к оттенкам серого и нормализация
image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Применение фильтра Гаусса
image = cv2.filter2D(image, ddepth=-1, kernel=gaussian)
if (save_images):
    cv2.imwrite('./images/gauss_blur.jpeg', image)

# Выделение границ при помощи оператора Собеля
gx = cv2.filter2D(image, ddepth=-1, kernel=sobelx)
gy = cv2.filter2D(image, ddepth=-1, kernel=sobely)

# Вычисление градиентов и углов в каждой точке
g = np.sqrt( np.power( gx, 2) + np.power( gy, 2 ) )
angle = np.degrees ( np.arctan2(gy, gx) )

if (save_images):
    cv2.imwrite('./images/sobel.jpeg', g)

# Аппроксимация углов
closest_dir = closest_dir_function(angle)
# Подавление не максимумов
thinned_output = non_maximal_suppressor(g, closest_dir)

if (save_images):
    cv2.imwrite('./images/thinned_output.jpeg', thinned_output*255)

output_img = thresholding(thinned_output)

if (save_images):
    cv2.imwrite('./images/canny.jpeg', output_img*255)