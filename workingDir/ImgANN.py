import matplotlib.pyplot as plt
from PIL import Image
from numpy import *
from pybrain.tools.xml import NetworkReader
from scipy import io
from scipy import ndimage
from sklearn.metrics import mean_squared_error



def plotData(image):
    # plots the input data '''
    # the image matrix will have to be transposed to be viewed correcty
    # cmap shows the color map
    plt.imshow(image.T, cmap='Greys')
    plt.show()

def getStatistics(x,y):
    predMSE = []
    predSSIM = []
    for j in range(len(Y)):
        if (Y[j] == p):
            predMSE.append(((mean_squared_error(X[j], X1)), j))  # mse val and index
            # print compare_ssim(X[j], X1)
    print "Mean Squared Error calculation: " + str(min(predMSE)) # Closer to 0, the better
    #print "Structural Similarity Index calculation: " + str(compare_ssim(x,y))


# load data
data = io.loadmat('ex4data1.mat')
size = (20, 20)

X = data['X']
Y = data['y']

# read test image
im = Image.open("images/1.png")
# convert to numpy array
if (len(shape(im)) == 3):
   imA = asarray(im, dtype="float")[:, :, 1]
else:
   imA = asarray(im, dtype="float")
# transform pixel values from 0 to 1 and invert and convert to PIL image
imA = (imA - amin(imA)) / (amax(imA) - amin(imA))
imA = 1 - imA
im1 = asarray(imA, dtype="float")
im1 = ndimage.grey_dilation(im1, size=(5, 5))
im1 = Image.fromarray(im1)
box = im1.getbbox()
im2 = im1.crop(box)
im3 = im2.resize(size)
im3 = asarray(im3, dtype="float")
im3 = 1 - im3.T
im3 = uint8(im3)
plotData(im3)
X1 = im3.reshape((X.shape[1]))


net = NetworkReader.readFrom('dig1.xml')
prediction = net.activate(X1)

p = argmax(prediction, axis=0)
print(prediction)
print("predicted output is \t" + str(p))
getStatistics(X1,p)


