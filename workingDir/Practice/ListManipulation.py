from sklearn import datasets
from PIL import Image, ImageChops
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
import cv2


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


digits = datasets.load_digits()

mseList = ([])
ssimList = ([])
imChopsDiff = ([])
ccList = ([])

ima5 = Image.open("images/2.png").resize((8,8)).convert('L')
image5 = np.array(ima5)

cv2Img = cv2.imread("images/2.png")
cv2.cvtColor(cv2Img, cv2.COLOR_BGR2GRAY)
#compare_ssim(cv2Img,digits.images[0]) #    Input images must have the same dtype
print digits.images.dtype
print image5.dtype


float(image5)
float(digits.image)

print image5.dtype
print digits.images.dtype

print digits.images[0].dtype
#ImageChops.difference(image5,digits.images[0])
i = 0
while i < len(digits.images):
    #mseList.append([i])
    mseList.append([mean_squared_error(image5,digits.images[i]), i])
    compare_ssim(image5,digits.images[i]) #    Input images must have the same dtype
    ccList.append((correlation_coefficient(image5, digits.images[i]),i))


    i = i+1

print sorted(mseList)

print ssimList

print sorted(ccList)



