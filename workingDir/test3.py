from PIL import Image
import numpy as np
from pybrain.datasets import ClassificationDataSet
i1 = Image.open("images/1.png").resize((8,8))
img1 = np.array(i1)

i2 = Image.open("images/2.png")
img2 = np.array(i2)

l= img2.flatten()
m=img1.flatten()

ds = ClassificationDataSet(1, 1)
ds.addSample((l,m), l)


