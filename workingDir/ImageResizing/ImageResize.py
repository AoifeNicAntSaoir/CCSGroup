####################For resizing images#######################################
#   Original image drawn by paint 256 x 256 pixels

from PIL import Image
from resizeimage import resizeimage
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

with open('testImage.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_cover(image, [8, 8])
        cover.save('test-image.png', image.format)




