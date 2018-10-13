import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol','Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines','Proline']

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =train_test_split(X, y,test_size=0.3,random_state=0,stratify=y)

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)

n_space = range(1,21)

random_accu_score = []
for n in n_space:
    forest.n_estimators = n
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    random_accu_score.append(metrics.accuracy_score(y_train, y_train_pred))
    
fig1 = plt.figure()
plt.plot(n_space, random_accu_score, label='Random Forest Train Set')
plt.legend(loc = 4)
plt.xlabel("N Values")
plt.ylabel("Accuracy Scores")
plt.xticks(range(0,21))
plt.title("In-sample Accuracy Scores VS. N Values")

#print(random_accu_score)
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))

fig2 = plt.figure()
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10,random_state=1).split(X_train,y_train)
scores = []
for k, (train, test) in enumerate(kfold):
    forest.fit(X_train[train], y_train[train])
    score = forest.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(y_train[train]), score))


print("My name is {Yue Liu}")
print("My NetID is: {yueliu6}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")










