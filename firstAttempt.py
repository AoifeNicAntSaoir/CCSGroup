from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from sklearn.datasets import load_digits
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
inLayer = LinearLayer(len(t))
hiddenLayer = SigmoidLayer(len(t))
outLayer = LinearLayer(1)

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




# daSet = SupervisedDataSet(len(t),1)




# for inpt, target in ds
# print inpy, target

# trainer = BackpropTrainer(net,daSet)

# trainer.train()
