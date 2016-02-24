import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

trainX = np.array([[0, -1], [0, .3], [.3, .3], [-1, -5], [.3, 0], [-1, 0 ], [1, 1], [2, 3]]);
trainY = np.array([-1, -1, -1, -1, 1, 1, 1, 1]);

testX1 = np.array([[-2, -2], [5,5], [0, 0], [-.5, .5], [.5, -.5]])
testX2 = np.array([[-1, -1], [2,2], [.5, 0], [-1, 2], [-.5, -1]])

#Train Naive Bayes Gaussian with test data
gaussLF = GaussianNB();
gaussLF.fit(trainX, trainY);
gaussY1 = gaussLF.predict(testX1);
print("gauss-only Y1", gaussY1);
gaussY2 = gaussLF.predict(testX2);
print("gauss-only Y2", gaussY2);

#Train Decision Tree with same test data
treeLF = DecisionTreeClassifier();
treeLF.fit(trainX, trainY);
treeY1 = treeLF.predict(testX1);
print("tree-only Y1", treeY1);
treeY2 = treeLF.predict(testX2);
print("tree-only Y2", treeY2);

#Now retrain the Gaussian with the training features and testX2 (using the Tree's predictions as training values)
gaussLF.fit(np.concatenate([trainX, testX2]), np.concatenate([trainY, treeY2]));
gaussY1WithX2TrainedTree = gaussLF.predict(testX1);
print("X2-tree-prediction-trained gauss Y1", gaussY1WithX2TrainedTree);

#Now retrain the Gaussian with the training features and testX1
gaussLF.fit(np.concatenate([trainX, testX1]), np.concatenate([trainY, treeY1]));
gaussY2WithX1TrainedTree = gaussLF.predict(testX2);
print("X1-tree-prediction-trained gauss Y2", gaussY1WithX2TrainedTree);

#Now similarly retrain the decision tree with the training features and testX2
treeLF.fit(np.concatenate([trainX, testX2]), np.concatenate([trainY, gaussY2]));
treeY1WithX2TrainedGauss = treeLF.predict(testX1);
print("X2-gaussian-trained tree Y1", treeY1WithX2TrainedGauss);

#Now similarly retrain the decision tree with the training features and testX1
treeLF.fit(np.concatenate([trainX, testX1]), np.concatenate([trainY, gaussY1]));
treeY2WithX1TrainedGauss = treeLF.predict(testX2);
print("X1-gaussian-trained tree Y2", treeY2WithX1TrainedGauss);

