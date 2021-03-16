import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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

knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=1, n_neighbors=5, p=1,
                               weights='uniform')

listOfClf = [knn_clf]

listOfPredictions = []
for clf in listOfClf:
    clf.fit(Xtr, Ytr)

    # Use this line for testing out the AUC Curve
    listOfPredictions.append(clf.predict_proba(Xva))

    # Use this line for writing to Kaggle
    # listOfPredictions.append(clf.predict_proba(Xte))

predictions = np.mean(np.array([listOfPredictions[0]]), axis=0)

false_positive_rate, true_positive_rate, thresholds = roc_curve(Yva, predictions[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print(roc_auc)

np.savetxt('kaggle2021_predictions.txt',
           np.vstack((np.arange(len(predictions)), predictions[:, 1])).T,
           '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')



