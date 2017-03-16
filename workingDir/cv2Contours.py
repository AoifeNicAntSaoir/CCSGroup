# Import the modules
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import numpy as np
from collections import Counter
import cv2

# Load the dataset
digits = datasets.load_digits()
#print(digits.keys())

#   Extracting features and images from load_digits Dataset
clf = LinearSVC()
clf.fit(digits.data, digits.target)
joblib.dump(clf, "digits_cls.pkl", compress=3)

#Read in Image
image = cv2.imread("photo_2.jpg")
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayImage = cv2.GaussianBlur(grayImage, (5,5), 0)
#  converts grayscale image into a binary image using a threshold value of 90.
# All the pixel locations with grayscale values greater than 90 are set to 0 in the binary image
# all the pixel locations with grayscale values less than 90 are set to 255 in the binary image.
ret, im_th = cv2.threshold(grayImage, 90, 255, cv2.THRESH_BINARY_INV)
#calculate the contours
im, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#for each bounding box, we generate a bounding square around each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
for rect in rects:
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3]*1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
    # Resize the image
    roi = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()