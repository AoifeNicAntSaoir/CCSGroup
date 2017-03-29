import matplotlib.pyplot as plt
from numpy import *
from pybrain.tools.xml import NetworkReader
from scipy import io

# load data
data = io.loadmat('data_mnist.mat')  # load the data

X = data['X']
Y = data['y']  # split to x and y

c = random.randint(0, X.shape[0])  # get random index

c2 = X[c, :] #  get the data stored at that index


#  show the in a graph
m, n = shape(X)
image = array(X[c, 0:n])
plt.imshow((image.reshape(20, 20)).T, cmap='Greys')
plt.show()

# read the saved network from the file
net = NetworkReader.readFrom('test_temp.xml')

# pass the test image through the neural net
prediction = net.activate(c2)
# get the value with the highest probability
p = argmax(prediction, axis=0)
print(prediction)
print("predicted output is \t" + str(p))
