from PIL import Image
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
import numpy as np
from sklearn import datasets


ima3 = np.array(Image.open("images/1.png").convert('L'))
ima4 = np.array(Image.open("images/2i.png").convert('L'))
print mean_squared_error(ima3, ima4)
print compare_ssim(ima3, ima4)


ima5 = Image.open("images/2ii.png").resize((8,8)).convert('L')
image5 = np.array(ima5)
print image5
print "\n "
ima6 = Image.open("images/2.png").resize((8,8)).convert('L')
image6 = np.array(ima6)
print image6

print compare_ssim(image5, image6)
print mean_squared_error(image5, image6)


