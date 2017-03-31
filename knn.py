from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

mnist = datasets.load_digits()

(train_data, test_data, train_labels, test_labels) = train_test_split(np.array(mnist.data),
                                        mnist.target, test_size=0.25, random_state=42)

(train_data,val_data, train_labels, val_labels) = train_test_split(train_data, train_labels,
                                test_size=0.1, random_state=84)

k_vals = range(1,30,2)
scores = []

for k in xrange(1,30,2):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores.append(knn.fit(train_data, train_labels).score(val_data, val_labels))

i = np.argmax(scores)
print("The highest accuracy is %.2f%%" % (scores[i] * 100))

model = KNeighborsClassifier(n_neighbors=k_vals[i])
model.fit(train_data, train_labels)
predictions = model.predict(test_data)


for i in np.random.randint(0, high=len(test_labels), size=(5,)):

    image = test_data[i]
    prediction = model.predict(image.reshape(1,-1) )[0]

    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=80, inter=cv2.INTER_CUBIC)

    print("Predicted digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)


fig = plt.figure(1)
plt.plot(k_vals, scores, 'ro', figure=fig)

fig.suptitle("Nearest Neighbor Classifier Accuracies")
fig.axes[0].set_xlabel("k (# of neighbors considered)")
fig.axes[0].set_ylabel("accuracy (% correct)");
fig.axes[0].axis([0, max(k_vals) + 1, 0, 1]);

plt.show()

