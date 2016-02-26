import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#options
n_samples = 400
cluster_std = 3

#Define a function for computing the accuracy of our methods
def computeAccuracy(actual, expected):
    maximum = len(expected);
    numCorrect = 0;
    for ii in range(maximum):
        if actual[ii] == expected[ii]:
            numCorrect = numCorrect + 1;
    return float(numCorrect)/float(maximum);

#Create two-feature, two-label data:
(features, labels) = make_blobs(centers=2, n_samples=n_samples, cluster_std=cluster_std);

#Let's pretend that trainX are the only labeled data...
trainBoundary = int(n_samples*.1)
testBoundary = int((n_samples - trainBoundary)/2)
trainX = features[0:trainBoundary]
trainY = labels[0:trainBoundary]
#...in which case, we'd split up the unlabeled data into two parts.
testX1 = features[trainBoundary: testBoundary]
testY1 = labels[trainBoundary: testBoundary]
testX2 = features[testBoundary: n_samples]
testY2 = labels[testBoundary: n_samples]

#Now let's try some semi-supervised learning.
#We'll use the official training data to train a Naive Bayes Gaussian model.
#After training the model, we use it to make guesses about testX1 and testX2.
gaussLF = GaussianNB();
gaussLF.fit(trainX, trainY);
gaussY1 = gaussLF.predict(testX1);
gaussY2 = gaussLF.predict(testX2);

#Now let's use the Gaussian model's guess about the testX1 dataset to train
#a Decision Tree model.
treeLF = DecisionTreeClassifier();
treeLF.fit(np.concatenate([trainX, testX1]), np.concatenate([trainY, gaussY1]));
treeY2 = treeLF.predict(testX2);
gaussY1Accuracy = "{:.2%}".format(computeAccuracy(gaussY1, testY1))
gaussY2Accuracy = "{:.2%}".format(computeAccuracy(gaussY2, testY2))
treeY2Accuracy = "{:.2%}".format(computeAccuracy(treeY2, testY2))
print("Gaussian's testX1 predictions were this accurate: ", gaussY1Accuracy);
print("Gaussian's testX2 predictions were this accurate: ", gaussY2Accuracy );
print("Decision Tree's testX2 predictions were this accurate: ", treeY2Accuracy);

#Graph the points for fun.
xs, ys = features.T
plt.plot(xs, ys, 'ro');
plt.show()
