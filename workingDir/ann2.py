from PIL import Image
import cv

img = Image.open('trolltunga.jpg').convert('L')
newImg = img.resize((8,8))
gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
cv.imwrite('graytest.jpg',gray)

