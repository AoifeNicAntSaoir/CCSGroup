import os
from random import randint, shuffle
import matplotlib.pyplot as plt
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader
from pybrain.utilities import percentError
from pylab import imshow
from scipy import io
import scipy
from numpy import *
import os
from PIL import Image
import numpy as np



i2 = Image.open("images/2.png")
i1 = Image.open("images/1.png")

img1 = np.array(i1).astype(float)
img2 = np.array(i2).astype(float)


ds = ClassificationDataSet(2,1)
ds.addSample((img2,1), 1)

