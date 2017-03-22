from sklearn.datasets import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.measure import compare_ssim as ssim
import math
from sklearn import svm

digits = load_digits()


def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

img1 = cv2.imread("8x8.png")
img2 = cv2.imread("8x8v2.png")

#grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fig = plt.figure()
fig.savefig('out.png')

mseval = mse(img1, img2)

#Structural Similarity Index (SSIM) algorithm -> models the perceived change
ssimval = ssim(img1, img2)

#Root mean squared error (RMSE) -> RMSE does not necessarily increase with
#                                  the variance of the errors. RMSE increases
#                                  with the variance of the frequency
#                                  distribution of error magnitudes.
rmse = math.sqrt(mse(img1,img2))

print ("MSE > ")
print(mseval)
print ("SSIM > ")
print (ssimval)


#print list of digits individually
fig = plt.figure()
for i in range(10):

    #plt.imshow(np.reshape(digits.data[i],(8,8)),cmap='gray')
    #fig = plt.figure(digits.data[i])

    subplot = fig.add_subplot(5,2,i+1)
    print(subplot,"fig_out{0}".format(i)+".png")

    #subplot.savefig("fig_out{0}".format(i)+".png")
    subplot.matshow(np.reshape(digits.data[i],(8,8)),cmap='gray')


#print first digit of dataset
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


for k in range(digits.images.__len__()):
    msevalDS = mse(img1,digits.images[k])
    list_msevals = []
    list_msevals += msevalDS
    print("MSE with Dataset > ")
    print(msevalDS)

    #print("list_msevals  > ")
    #print(list_msevals)