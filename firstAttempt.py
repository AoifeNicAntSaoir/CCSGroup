from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
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

# net = buildNetwork(len(t), len(t), 1)

#initialize a feed foward network
net = FeedForwardNetwork()

#create layers for FFN
inLayer = LinearLayer(len(t)) #sets up the number of nodes based on 'length' of the loaded image
hiddenLayer = SigmoidLayer(len(t))
outLayer = LinearLayer(10)#you need ten outputs - one for each digit(0,1,2,3 etc)

# add layers to FFN
net.addInputModule(inLayer)
net.addModule(hiddenLayer)
net.addOutputModule(outLayer)

#create connections between the layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)
#add connections
net.addConnection(in_to_hidden)
net.addConnection(hidden_to_out)

net.sortModules()

print net


digits = load_digits()
X,y = digits.data, digits.target



print (X.shape)


plt.gray()
plt.matshow(digits.images[2])
plt.show()



daSet = ClassificationDataSet(len(t),1)
for k in xrange(len(X)):
    daSet.addSample(X.ravel()[k],y.ravel()[k])


testData, trainData = daSet.splitWithProportion(0.25)


trainData._convertToOneOfMany( )
testData._convertToOneOfMany( )
#for inpt, target in daSet:
   # print inpt, target

trainer = BackpropTrainer(net, dataset=trainData, momentum=0.1, learningrate=0.01, verbose=True)

trainer.trainEpochs(50)
print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
           dataset=testData )
           , testData['class'] )

trainer.train()
