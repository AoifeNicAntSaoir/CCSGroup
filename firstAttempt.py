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

digits = load_digits()
X, y = digits.data, digits.target
# net = buildNetwork(len(t), len(t), 1)
daSet = ClassificationDataSet(len(t), 1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k], y.ravel()[k])

testData, trainData = daSet.splitWithProportion(0.25)

trainData._convertToOneOfMany()
testData._convertToOneOfMany()
# initialize a feed foward network

if os.path.isfile('dig.xml'):
    net = NetworkReader.readFrom('dig.xml')
else:
    # net = FeedForwardNetwork()
    net = buildNetwork(trainData.indim, 64, trainData.outdim, outclass=SoftmaxLayer)
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

# net.sortModules()

print net

print (X.shape)

plt.gray()
plt.matshow(digits.images[2])
plt.show()

# for inpt, target in daSet:
# print inpt, target

trainer = BackpropTrainer(net, dataset=trainData, momentum=0.1, learningrate=0.01, verbose=True)

trainer.trainEpochs(200)
print 'Percent Accuracy Test dataset: ', percentError(trainer.testOnClassData(
    dataset=testData)
    , testData['class'])

trainer.train()

NetworkWriter.writeToFile(net, 'dig.xml')
