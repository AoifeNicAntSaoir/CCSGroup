from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.utilities import percentError
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from random import randint, shuffle
from pylab import imshow
from scipy import io, ndimage
from numpy import *
import os
import matplotlib.pyplot as plt


def plot_the_num(X, Y, ca):
    # plots the given image
    # the image matrix is transposed
    # color set to grey
    input_image = X[ca, :]
    imshow((input_image.reshape(20, 20)).T, cmap='Greys')

    # plot the same output case
    print('the digit printed is', Y[ca][0])


def convert_to_one_of_many(Y):
    # converts Y to one of many types
    # or one output per label
    rows, cols = Y.shape
    classes = unique(Y).size  # should get 10 classes
    newY = zeros((rows, classes))

    for i in range(0, rows):
        newY[i, Y[i]] = 1

    return newY


# load the MNIST data
digits = io.loadmat('ex4data1.mat')

# making X and Y numpy arrays
X = digits['X']
Y = digits['y']

Y[Y == 10] = 0  # 0 has the 10th position, this line gives it the 0th position

num_of_labels = unique(Y).size  # gets your 10 labels

print('show a random number from the MNIST database')
random_ind = randint(0, X.shape[0])  # gets a random number between 0 and the size of the X array
plot_the_num(X, Y, random_ind)
plt.show()

# build the dataset
X = c_[ones(X.shape[0]), X]
num_of_examples, size_of_example = X.shape
# convert the test data to one of many (10)
Y = convert_to_one_of_many(Y)

# seperating training and test datasets

X1 = hstack((X, Y))
shuffle(X1)

X = X1[:, 0:size_of_example]
Y = X1[:, size_of_example: X1.shape[1]]

# add the contents of digits to a dataset
# daSet = ClassificationDataSet(sizeOfExample, numOfLabels)
# for k in xrange(len(X)):
#    daSet.addSample(X.ravel()[k], Y.ravel()[k])

# testData, trainData = daSet.splitWithProportion(0.25)

train_data = ClassificationDataSet(size_of_example, num_of_labels)
test_data = ClassificationDataSet(size_of_example, num_of_labels)

cutoff = int(num_of_examples * 0.7)

for i in range(0, cutoff):
    train_data.addSample(X[i, :], Y[i, :])

# setting the field names
train_data.setField('input', X[0:cutoff, :])
train_data.setField('target', Y[0:cutoff, :])

for i in range(cutoff, num_of_examples):
    test_data.addSample(X[i, :], Y[i, :])

test_data.setField('input', X[cutoff:num_of_examples, :])
test_data.setField('target', Y[cutoff:num_of_examples, :])

# split the dataset into training and testing
# testData, trainData = daSet.splitWithProportion(0.25)

# convert the data into 10 separate digits
# trainData._convertToOneOfMany()
# testData._convertToOneOfMany()

if os.path.isfile('dig.xml'):
    net = NetworkReader.readFrom('dig.xml')
    net.sorted = False
    net.sortModules()
else:

    net = buildNetwork(size_of_example,size_of_example/2,num_of_labels, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)

    # net.sorted = False
    net.sortModules()

test_index = randint(0, X.shape[0])
test_input = X[test_index, :]

print("Number to predict is", Y[random_ind][0])

real_train = train_data['target'].argmax(axis=1)
real_test = test_data['target'].argmax(axis=1)

EPOCHS = 5

trainer = BackpropTrainer(net, dataset=train_data, momentum=0.3, learningrate=0.01, verbose=False)

for i in range(EPOCHS):
    # set the epochs
    trainer.trainEpochs(1)

    outputTrain = net.activateOnDataset(train_data)
    outputTrain = outputTrain.argmax(axis=1)
    trainResult = percentError(outputTrain, real_train)

    outputTest = net.activateOnDataset(test_data)
    outputTest = outputTest.argmax(axis=1)
    testResult = percentError(outputTest, real_test)

    print('training set accuracy:', 100 - trainResult, 'test set accuracy:', 100 - testResult)

prediction = net.activate(test_input)
# returns the index of the highest value down the columns
p = argmax(prediction, axis=0)

NetworkWriter.writeToFile(net, 'dig.xml')

# plotData(X[:, 0:sizeOfExample-1], Y, randomIndex)
print("predicted output after training is", p)
