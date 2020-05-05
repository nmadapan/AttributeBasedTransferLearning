import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

#Plot Confusion Matrix
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
temp = np.logical_or(y == 0, y == 1)
y = y[temp]
X = X[temp, :]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state = 42)

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
# C = 1.0  # SVM regularization parameter
# models = (svm.SVC(kernel='linear', C=C),
#           svm.LinearSVC(C=C, max_iter=10000),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C),
#           svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
# models = (clf.fit(X, y) for clf in models)

C = 1.0
obj = svm.LinearSVC(C=C, class_weight = 'balanced')
obj.fit(x_train, y_train)

y_pred = obj.predict(x_test)
print(obj.decision_function(x_test))
print(y_pred)

cm = confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = [i for i in np.unique(y)],
                  columns = [i for i in np.unique(y)])
plt.figure(figsize = (5,5))
sn.heatmap(df_cm, annot=True)
plt.show()