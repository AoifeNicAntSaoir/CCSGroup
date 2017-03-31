import numpy as np

#               hours sleeping, hours studying,
X = np.array( ([3,5],
               [5,1],
               [10,2]), dtype=float)
#   y = Test Score
y = np.array( ([75],
               [82],
               [93]), dtype=float )

X = X/np.amax(X, axis=0)
y = y/100

print X.shape
print y.shape

class NeuralNetwork(object):
    def __init__(self):
        self.inputLayerSize= 2
        self.outputLayerSize= 1
        self.hiddenLayerSize= 3

    def forward(self, X):
        # propagate inputs through network

    def sigmoid(self, z):
        return  1/(1+np.exp(-z))

NN = NeuralNetwork
testInput = np.arange(-6,6,0.01)
plot(testInput, NN.sigmoid(testInput), lineWidth = 2)
grid = 1

NN.sigmoid(1)
NN.sigmoid(np.array([-1,0,1]))
NN.sigmoid(np.random.rand(3,3))

yHat = NN.forward(X)
print yHat
print y

bar([0,1,2]), y, width=0.35, alpha=0.8
bar([0.35, 1.35, 2.35], yHat, width=0.35, color='r', alpha=0.8)
grid(1)
legend(['y','yHat'])
