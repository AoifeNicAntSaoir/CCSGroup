from random import randint
from scipy import io
from numpy import unique, argmax
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pylab import imshow
from sklearn.datasets import load_digits
from numpy import unique, c_, ones, zeros, hstack, argmax
import matplotlib.pyplot as plt




def plotTheImage(X, y, c):
    inpImg = X[c, :]
    imshow((inpImg.reshape(20, 20)).T, cmap='Greys')

    print ['the digit printed is', y[c]]

def convertToOneOfMany(Y): # look and see if pyBrain has the functionality for this
    '''converts Y to One of many type '''

    rows, cols = Y.shape
    classes = unique(Y).size
    Y1 = zeros((rows, classes))

    for i in range(0, rows):
        Y1[i, Y[i]] = 1

    return Y1

data = io.loadmat('ex4data1.mat')

# making X and Y numpy arrays
X = data['X']
y = data['y']

# changing identity of digit '0' from 10 to 0 in array
y[y == 10] = 0
numOfLabels = unique(y).size

print('plotting a random digit from the input')
randomIndex = randint(0, X.shape[0])
plotTheImage(X, y, randomIndex)
plt.show()

X = c_[ones(X.shape[0]), X]
numOfExamples, sizeOfExample = X.shape
y= convertToOneOfMany(y)



dSet = ClassificationDataSet(sizeOfExample, numOfLabels)
for k in xrange(len(X)):
        dSet.addSample(X.ravel()[k], y.ravel()[k])

testData, trainData = dSet.splitWithProportion(0.25)
# numOfExamples, sizeOfExample = X.shape

trainData._convertToOneOfMany()
testData._convertToOneOfMany()

inputLayerSize = trainData.indim
hiddenLayerSize0 = trainData.indim
outputLayerSize = numOfLabels

inputLayer = LinearLayer(inputLayerSize)
hiddenLayer0 = SigmoidLayer(hiddenLayerSize0)
outputLayer = SoftmaxLayer(outputLayerSize)

ffNetwork = FeedForwardNetwork()

# adding the layers to the network
ffNetwork.addInputModule(inputLayer)
ffNetwork.addModule(hiddenLayer0)
ffNetwork.addOutputModule(outputLayer)

# initializing the thetas
theta1 = FullConnection(inputLayer, hiddenLayer0)
theta2 = FullConnection(hiddenLayer0, outputLayer)

# connecting the layers with thetas
ffNetwork.addConnection(theta1)
ffNetwork.addConnection(theta2)


# this sets the network
# input_layer->theta1->hidden_layer->theta2->output_layer
ffNetwork.sortModules()


print('_______testing without training________')

testIndex = randint(0, X.shape[0])
testInput = X[testIndex, :]
print X.shape
prediction = ffNetwork.activate(testInput)
p = argmax(prediction, axis=0)

print("true output is", y[randomIndex][0])
print("predicted output is", p)

print('_______testing after training_______')

trueTrain = trainData['target'].argmax(axis=1)
trueTest = testData['target'].argmax(axis=1)

EPOCHS = 5

trainer = BackpropTrainer(ffNetwork, dataset=trainData, verbose=False)

for i in range(EPOCHS):
    trainer.trainEpochs(1)

    # calculatig the error percentage
    outputTrain = ffNetwork.activateOnDataset(trainData)
    outputTrain = outputTrain.argmax(axis=1)
    trainResult = percentError(outputTrain, trueTrain)

    outputTest = ffNetwork.activateOnDataset(testData)
    outputTest = outputTest.argmax(axis=1)
    testResult = percentError(outputTest, trueTest)

    print('training set accuracy:', 100 - trainResult, 'test set accuracy:', 100 - testResult)

prediction = ffNetwork.activate(testInput)
p = argmax(prediction, axis=0)

# plotData(X[:, 0:sizeOfExample-1], Y, randomIndex)
print("predicted output after training is", p)