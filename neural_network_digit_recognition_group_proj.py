from random import randint, shuffle
import os
from numpy import *
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml import NetworkReader
from pybrain.tools.xml import NetworkWriter
from pybrain.utilities import percentError
from scipy import io
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2


def plotData(image):
	# plots the input data '''
	# the image matrix will have to be transposed to be viewed correctly
	# cmap shows the color map
	plt.imshow(image.T, cmap='Greys')
	plt.show()


def convert_to_one_of_many(y):
	# converts Y to one of many types
	# or one output per label
	rows, cols = y.shape
	classes = unique(y).size  # should get 10 classes
	new_y = zeros((rows, classes))

	for i in range(0, rows):
		new_y[i, y[i]] = 1

	return new_y


# load the MNIST data
digits = io.loadmat('data_mnist.mat')


# making X and Y numpy arrays
X = digits['X']
Y = digits['y']


Y[Y == 10] = 0  # 0 has the 10th position, this line gives it the 0th position


num_of_labels = unique(Y).size  # gets your 10 labels/outputs

# build the dataset

num_of_examples, size_of_example = X.shape

# convert the test data to one of many (10)
Y = convert_to_one_of_many(Y) # splits the data into 10 outputs

# separating training and test data sets

X1 = hstack((X, Y))  # puts into a single one dimensional array
shuffle(X1)  # shuffles the data

X = X1[:, 0:size_of_example]
Y = X1[:, size_of_example: X1.shape[1]]

# add the contents of digits to a dataset

train_data = SupervisedDataSet(size_of_example, num_of_labels)
test_data = SupervisedDataSet(size_of_example, num_of_labels)

data_split = int(num_of_examples * 0.9)

for i in range(0, data_split):
	train_data.addSample(X[i, :], Y[i, :]) #add .9 of the data to the ds

# setting the field names
train_data.setField('input', X[0:data_split, :])
train_data.setField('target', Y[0:data_split, :])

for i in range(data_split, num_of_examples):
	test_data.addSample(X[i, :], Y[i, :])

test_data.setField('input', X[data_split:num_of_examples, :])
test_data.setField('target', Y[data_split:num_of_examples, :])

#if os.path.isfile('extra_layers.xml'):
	#net = NetworkReader.readFrom('extra_layers.xml')

#else:
net = buildNetwork(size_of_example, int(size_of_example), int(size_of_example) / 5, num_of_labels,
					   hiddenclass=SigmoidLayer,
					   outclass=SoftmaxLayer)

net.sortModules()

test_index = randint(0, X.shape[0])
test_input = X[test_index]

real_train = train_data['target'].argmax(axis=1)
real_test = test_data['target'].argmax(axis=1)

EPOCHS = 3

trainer = BackpropTrainer(net, dataset=train_data, momentum=0.41, learningrate=0.03, verbose=False)

trainResultArr = []
epochs = []
testResultArr = []

for i in range(EPOCHS):
	# set the epochs
	trainer.trainEpochs(1)

	outputTrain = net.activateOnDataset(train_data)
	outputTrain = outputTrain.argmax(axis=1) # takes the output with the largest prob
	trainResult = percentError(outputTrain, real_train)

	outputTest = net.activateOnDataset(test_data)
	outputTest = outputTest.argmax(axis=1)
	testResult = percentError(outputTest, real_test)

	finalTrainResult = 100 - trainResult
	finalTestResult = 100 - testResult

	print "Epoch: " + str(i + 1) + "\tTraining set accuracy: " + str(finalTrainResult) + "\tTest set accuracy: " + str(
		finalTestResult)
	# getStatistics(	)

	trainResultArr.append(finalTestResult)
	testResultArr.append(finalTrainResult)
	epochs.append(i)

prediction = net.activate(test_input)

# returns the index of the highest value down the columns
p = argmax(prediction, axis=0)

NetworkWriter.writeToFile(net, 'extra_layers.xml')

plt.plot(epochs, trainResultArr)
plt.plot(epochs, testResultArr)
plt.title('Training Result (Orange) vs Test Result of ANN (Blue)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')

plt.show()

