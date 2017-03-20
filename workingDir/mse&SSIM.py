from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
print "\t\t\tImage 0 from the dataset\n =====================================\n"
print digits.images[0]

ima3 = np.array(Image.open("images/1.png").convert('L'))
ima4 = np.array(Image.open("images/2i.png").convert('L'))


ima5 = Image.open("images/2ii.png").resize((8,8)).convert('L')
image5 = np.array(ima5)
print "\n\n\t\t\tImage (png) read in\n =====================================\n"
print image5
print "\n "
ima6 = Image.open("images/2.png").resize((8,8)).convert('L')
image6 = np.array(ima6)
print image6

print digits.images[0].dtype
print image5.dtype
print mean_squared_error(ima3, ima4)
print compare_ssim(ima3, ima4)

im = image6.astype(float)

print im

print compare_ssim(im, digits.images[670])
print digits.keys()
print digits.target
print digits.target_names
