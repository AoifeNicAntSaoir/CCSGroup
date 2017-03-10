from skimage import data, io, filters
import cv2;


open("testImage.png")
img = cv2.imread('testImage.png')
print img
edges = filters.sobel(img)