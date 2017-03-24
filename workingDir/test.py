from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.datasets import fetch_mldata
import numpy as np
from PIL import Image

myImg = Image.open("images/1.png")
ig = np.array(myImg)


digits = fetch_mldata("MNIST original")
X = digits.data



images = digits.data[0].flatten()

print images
lenImg =  len(images)

print digits.keys()
print digits.data
print digits.target
print digits


net = buildNetwork(lenImg,5,2)

net.activate([1,1,0])

ds = SupervisedDataSet(3,2)
ds.addSample((0,0,0),(0))
ds.addSample((0,0,1),(0))
ds.addSample((0,1,0),(1))
ds.addSample((1,0,0),(0))
ds.addSample((0,1,1),(1))

trainer = BackpropTrainer(net, ds)
#trainer.trainUntilConvergence()

value = trainer.train()
print value