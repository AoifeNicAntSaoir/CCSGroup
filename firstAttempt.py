from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
NUM_EPOCHS = 50

# read image with cv2
def loadImage(path):
    im = cv2.imread(path)
    return flatten(im)


# flatten the image
def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


# pass image to store image a store it as t
t = loadImage('shanesC.png')
#load the data and store
digits = load_digits()

#set y as target
X, y = digits.data, digits.target


#add the contents of digits to a dataset
daSet = ClassificationDataSet(len(t), 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

#split the dataset into training and testing
testData, trainData = daSet.splitWithProportion(0.25)

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
    net = buildNetwork(trainData.indim, 155, 10,hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

# create a backprop trainer
trainer = BackpropTrainer(net, dataset=trainData, momentum=0.3, learningrate=0.3,weightdecay= 0.01, verbose=True)


################# this is from an old iteration will delete	
	
# create layers for FFN
# inLayer = LinearLayer(len(t)) #sets up the number of nodes based on 'length' of the loaded image
# hiddenLayer = SigmoidLayer(len(t))
# outLayer = LinearLayer(10)#you need ten outputs - one for each digit(0,1,2,3 etc)

# add layers to FFN
# net.addInputModule(inLayer)
# net.addModule(hiddenLayer)
# net.addOutputModule(outLayer)

# create connections between the layers
# in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_out = FullConnection(hiddenLayer, outLayer)
# add connections
# net.addConnection(in_to_hidden)
# net.addConnection(hidden_to_out)




#a test to show the digits in the dataset, try changing the 2 and it will blwo your mind
"""plt.gray()
plt.matshow(digits.images[2])
plt.show()"""




#set the epochs
trainer.trainEpochs(2000)
NetworkWriter.writeToFile(net, 'dig.xml')

#print results
print 'Percent Error dataset: ', percentError(trainer.testOnClassData(
    dataset=testData)
    , testData['class'])

exit(0)

