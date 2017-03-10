from sklearn import datasets

digits = datasets.load_digits()

#print datasets.load_digits().keys()

#images_and_labels = list(zip(digits.images, digits.target))
#print images_and_labels

print(digits.data)
print("\n\n\n\n\n")
print(digits.keys())
print("\n\n\n\n\n")
print(digits.target)
print("\n\n\n\n\n")
print(digits.DESCR)

# Import matplotlib
import matplotlib.pyplot as plt

# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

print digits.images[0]

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))



# Show the plot
plt.show()

