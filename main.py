import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn import gaussian_process
from sklearn import linear_model

X = np.genfromtxt("data/X_train.txt")
Y = np.genfromtxt("data/Y_train.txt")
X, Y = ml.shuffleData(X, Y)
Xte = np.genfromtxt("data/X_test.txt")

Xtr, Xva, Ytr, Yva = ml.splitData(X, Y, 0.75)

# Random Forest Classifier       def=100      def=auto          def=none
rfC = RandomForestClassifier(n_estimators=100, max_features=4, max_depth=20)

# KNN classifier             def=2
knnC = KNeighborsClassifier(p=1)

# AdaBoost Classifier
# decTree = tree.DecisionTreeClassifier(max_depth=3)
# adaC = AdaBoostClassifier(base_estimator=decTree, n_estimators=50)

# Gradient Boosting
gradC = GradientBoostingClassifier(max_depth=10, n_estimators=750, max_features=4)

listOfC = [rfC, knnC, gradC]

listOfPredictionsAUC = []
listOfPredictionsKaggle = []
for c in listOfC:
    c.fit(Xtr, Ytr)

    # Use this line for testing out the AUC Curve
    listOfPredictionsAUC.append(c.predict_proba(Xva))

    # Use this line for writing to Kaggle
    listOfPredictionsKaggle.append(c.predict_proba(Xte))

predictionsAUC = np.mean(np.array([listOfPredictionsAUC[0], listOfPredictionsAUC[1], listOfPredictionsAUC[2]]), axis=0)
predictionsKaggle = np.mean(np.array([listOfPredictionsKaggle[0], listOfPredictionsKaggle[1], listOfPredictionsKaggle[2]]), axis=0)

# get the auc data
false_positive_rate, true_positive_rate, thresholds = roc_curve(Yva, predictionsAUC[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

np.savetxt('kaggle2021_predictions.txt',
           np.vstack((np.arange(len(predictionsKaggle)), predictionsKaggle[:, 1])).T,
           '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')



