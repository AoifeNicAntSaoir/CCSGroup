from random import randint, shuffle

from numpy import *
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml import NetworkReader
from pybrain.tools.xml import NetworkWriter
from pybrain.utilities import percentError
from scipy import io


def convert_to_one_of_many(Y):
    # converts Y to one of many types
    # or one output per label
    rows, cols = Y.shape
    classes = unique(Y).size  # should get 10 classes
    new_y = zeros((rows, classes))

    for i in range(0, rows):
        new_y[i, Y[i]] = 1

    return new_y


# load the MNIST data
digits = io.loadmat('data_mnist.mat')

# making X and Y numpy arrays
X = digits['X']

Y = digits['y']

Y[Y == 10] = 0  # 0 has the 10th position, this line gives it the 0th position

num_of_labels = unique(Y).size  # gets your 10 labels/outputs


# build the dataset

# X = c_[ones(X.shape[0]), X]

num_of_examples, size_of_example = X.shape
# convert the test data to one of many (10)
Y = convert_to_one_of_many(Y)

# separating training and test data sets

X1 = hstack((X, Y))  # puts into a single one dimensional array
shuffle(X1)  # shuffles the data

X = X1[:, 0:size_of_example]
Y = X1[:, size_of_example: X1.shape[1]]

# add the contents of digits to a dataset
# daSet = ClassificationDataSet(sizeOfExample, numOfLabels)
# for k in xrange(len(X)):
#    daSet.addSample(X.ravel()[k], Y.ravel()[k])

# testData, trainData = daSet.splitWithProportion(0.25)

train_data = ClassificationDataSet(size_of_example, num_of_labels)
test_data = ClassificationDataSet(size_of_example, num_of_labels)

data_split = int(num_of_examples * 0.7)

for i in range(0, data_split):
    train_data.addSample(X[i, :], Y[i, :])

# setting the field names
train_data.setField('input', X[0:data_split, :])
train_data.setField('target', Y[0:data_split, :])

# print train_data.getField(X[0:cutoff, :])

for i in range(data_split, num_of_examples):
    test_data.addSample(X[i, :], Y[i, :])

test_data.setField('input', X[data_split:num_of_examples, :])
test_data.setField('target', Y[data_split:num_of_examples, :])

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

    net = buildNetwork(size_of_example, 250, num_of_labels, hiddenclass=SigmoidLayer,
                       outclass=SoftmaxLayer)

    # net.sorted = False
    net.sortModules()

test_index = randint(0, X.shape[0])
test_input = X[test_index]


real_train = train_data['target'].argmax(axis=1)
real_test = test_data['target'].argmax(axis=1)

EPOCHS = 14

trainer = BackpropTrainer(net, dataset=train_data, momentum=0.3, learningrate=0.06, verbose=False)

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
