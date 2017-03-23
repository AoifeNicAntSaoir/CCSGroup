from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from skimage.measure import compare_ssim
from sklearn.datasets import fetch_mldata, load_digits
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from sklearn.metrics import mean_squared_error
import os
import numpy as np
from PIL import Image, ImageChops
NUM_EPOCHS = 50
from sklearn import datasets
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


# open image resize & grayscale
ima5 = Image.open("images/2.png").resize((25, 25)).convert('L')
image5 = np.array(ima5).astype(float)


#load the data and store
digits = fetch_mldata("MNIST original")

#print ImageChops.difference(ima5, ima5).getbbox()

#set up lists
mseList = ([])
ssimList = ([])
imChopsDiff = ([])
ccList = ([])

# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension

print mean_squared_error(digits.data[0], digits.data[1])

i = 0
while i < len(digits.data):
    #mseList.append([i])
    mseList.append([mean_squared_error(image5,digits.data[i]), i])
    compare_ssim(image5,digits.data[i]) #    Input images must have the same dtype
    ccList.append((correlation_coefficient(image5, digits.data[i]),i))
    i = i+1
print mseList
print sorted(mseList)

print ssimList

print sorted(ccList)

"""



"""
#set y as target
X, y = digits.data, digits.target


#add the contents of digits to a dataset
daSet = ClassificationDataSet(64, 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

#split the dataset into training and testing
testData, trainData = daSet.splitWithProportion(0.40)

#convert the data into 10 separate digits
trainData._convertToOneOfMany()
testData._convertToOneOfMany()


#check for the save file and load
if os.path.isfile('dig.xml'):
    net = NetworkReader.readFrom('dig.xml')
    net.sorted = False
    net.sortModules()
else:
    # net = FeedForwardNetwork()
    net = buildNetwork(64, 37,10, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer, bias=True)

# create a backprop trainer
trainer = BackpropTrainer(net, dataset=trainData, momentum=0.0, learningrate=0.01,weightdecay= 0.01, verbose=True)

i = 0
trainer.trainUntilConvergence()
print(trainData.indim)
print(testData.indim)

NetworkWriter.writeToFile(net, 'dig.xml')


exit(0)