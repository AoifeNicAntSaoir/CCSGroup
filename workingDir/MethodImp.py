import math
import numpy as np
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

print sigmoid(0.8)

def multiply(neuron, weight):
    return neuron * weight


input1 = 1
input2 = 1

target = 0

w1 = 0.8
w2 = 0.4
w3 = 0.3

w4 = 0.2
w5 = 0.9
w6 = 0.5


preSigH1 = (input1 * w1) + (input2 * w4)
preSigH2 = (input1 * w2) + (input2 * w5)
preSigH3 = (input1 * w3) + (input2 * w6)

h1 = sigmoid(preSigH1)
h2 = sigmoid(preSigH2)
h3 = sigmoid(preSigH3)

w7 = 0.3
w8 = 0.5
w9 = 0.9

preSigOutput = ((h1 * w7) + ( h2 * w8) + ( h3 * w9))
print preSigOutput

output = sigmoid(preSigOutput)
print output

MarginOfError = target - output
print "Margin of Error: " +  str(MarginOfError)

#############################################
DeltaOutputSum = output * MarginOfError
print DeltaOutputSum
DeltaWeights = DeltaOutputSum * np.array([h1, h2 ,h3])
print DeltaWeights


np.array
