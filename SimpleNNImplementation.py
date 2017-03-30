import numpy as np
from sklearn.datasets import fetch_mldata
from PIL import Image

ds = fetch_mldata("MNIST Original")
X = ds.data
Y = ds.target

X = np.append(X,Y)
print X.shape

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#myImg = Image.open("images/1.png").resize((20,20))
# input dataset


# output dataset
y = np.array(Y).T
print y

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3), (2)) - 1

for iter in xrange(10000):
    # forward propagation
    l0 = X


    l1 = sigmoid(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print "Output After Training:"
print l1
