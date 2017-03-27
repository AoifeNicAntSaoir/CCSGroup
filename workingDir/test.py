import cv2
im_gray = cv2.imread('images/blue.png', cv2.IMREAD_GRAYSCALE)
thresh = 127
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('images/blackwhite.png', im_bw)