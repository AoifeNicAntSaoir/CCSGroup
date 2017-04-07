import matplotlib.pyplot as plt
from numpy import *
from pybrain.tools.xml import NetworkReader
from scipy import io

# load data
data = io.loadmat('data_mnist.mat')  # load the data

X = data['X']  # store the parts of the data labeled X so that we can get its size and shape

c = random.randint(0, X.shape[0])  # get random index

c2 = X[c, :]  # get the data stored at that index

#  show the digit in a graph
m, n = shape(X)
image = array(X[c, 0:n])
plt.imshow((image.reshape(20, 20)).T, cmap='Greys')
plt.show()

# read the saved network from the file
net = NetworkReader.readFrom('good_net.xml')

# pass the test image through the neural net
prediction = net.activate(c2)
# get the value with the highest probability
p = argmax(prediction, axis=0)
print(prediction)
print("The digit should be : \t" + str(p))
