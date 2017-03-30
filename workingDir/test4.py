  # Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
#Logistic regression is a statistical method for analyzing a dataset in which there are one or more
# independent variables that determine an outcome. The outcome is measured with a dichotomous
# variable (in which there are only two possible outcomes).

digits = datasets.load_digits()

model = LogisticRegression()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data.astype(np.float64), digits.target.astype(np.float64), test_size=0.4, train_size=0.6)

def sigmoid(x):
    return 1/1+np.exp(x)

print sigmoid(2)

def tanh(x):
    return np.tanh(2)

model.fit(Xtrain,Ytrain)

expected = Ytest
predicted = model.predict(Xtest)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

