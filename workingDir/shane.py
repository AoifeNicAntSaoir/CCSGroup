from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import os
from skimage import img_as_int, img_as_uint, img_as_bool, img_as_float
from PIL import Image
import numpy as np
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import cv2

NUM_EPOCHS = 50

dataset = datasets.fetch_mldata("MNIST Original")
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
# read image with PIL.Image & load it, resize, convert to grayscale & load it into an array
img = Image.open("images/2ii.png").resize((8,8)).convert('L')
image = np.array(img)
clf = LinearSVC()
clf.fit(hog_features, labels)
#print img_as_int(image)
#print img_as_uint(image)
#load the data and store
digits = load_digits()

#set y as target output
X, y = digits.images , digits.target

image,ctrs, hier = cv2.findContours(image,)

ima5 = Image.open("images/3.png").resize((28,28)).convert('L')
image5 = np.array(ima5)
#add the contents of digits to a dataset for supervised classification training
daSet = ClassificationDataSet(64, 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

print "digits.images[0]"
d = digits.images[0]
print d

print img_as_float(image)
#print img_as_int(d)
#split the dataset into training and testing
testData, trainData = daSet.splitWithProportion(0.40)

#convert the data into 10 separate digits
trainData._convertToOneOfMany()
testData._convertToOneOfMany()

#already has inputs and connections/synapses
net = buildNetwork(64, 37,10, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer, bias=True)

#for inpt, target in daSet:
#    print target, inpt

# create a backprop trainer
trainer = BackpropTrainer(net, dataset=trainData, momentum=0.1, verbose=True, weightdecay= 0.01)
trainer.trainUntilConvergence()
print(trainData.indim)
print(testData.indim)

#set the epochs
#trainer.trainEpochs(5)
NetworkWriter.writeToFile(net, 'dig.xml')

trainer.trainEpochs (50)
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
           dataset=testData )
           , testData['class'] )


#print net.activate(t)


#print results
#print 'Percent Error dataset: ', percentError(trainer.testOnClassData(
#    dataset=testData)
#    , testData['class'])

exit(0)