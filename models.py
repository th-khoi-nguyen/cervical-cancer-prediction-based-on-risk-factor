import pandas as pd
import numpy as np

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

df = pd.read_csv('/content/drive/MyDrive/ML Projects/cervical_cancer/new_risk_factor_cervical_cancer.csv')

# use Schiller's values as labels to predict
y = df['Schiller']
x = df.drop(columns=['Schiller'])

from sklearn.model_selection import KFold 
kf = KFold(n_splits = 5, shuffle = True, random_state = 1) # define the split - split into 5 folds with data shuffle 
folds = kf.split(x)

mses = []
score = []
precision = []
accuracy = []
recall = []
results = []
for train_index, test_index in folds:
    x_train = x.iloc[train_index]
    y_train = y.iloc[train_index]
    x_test = x.iloc[test_index]
    y_test = y.iloc[test_index]

# test the performance of each k value
for k in range(2, 11):                            
    model = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score.append(model.score(x_test, y_test))
    precision.append(precision_score(y_test, y_pred, average='macro'))  
    accuracy.append(accuracy_score(y_test, y_pred))
    recall.append(precision_score(y_test, y_pred, average='macro'))
    mses.append (mean_squared_error(y_test, y_pred))
    results.append({"k":k, "score": (sum(score)/len(score))})
    
sorted(results, key=lambda d: d['score'], reverse=True) # decide the best k by accuracy  

def classifier (i):
    folds = kf.split(x)
    precision = []
    accuracy = []
    recall = []
    for train_index, test_index in folds:
        x_train = x.iloc[train_index]
        y_train = y.iloc[train_index]
        x_test = x.iloc[test_index]
        y_test = y.iloc[test_index]
        if i == 'knn': 
            model = KNeighborsClassifier(n_neighbors=2).fit(x_train, y_train)
        if i == 'NB':
            model = GaussianNB().fit(x_train, y_train)
        if i == 'LoG':
            model = LogisticRegression(random_state=1).fit(x_train, y_train)
        if i == 'DT':
            model = DecisionTreeClassifier(random_state=1).fit(x_train, y_train)
        if i == 'RF':
            model = RandomForestClassifier(random_state=1).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        precision.append(precision_score(y_test, y_pred, average='macro'))  
        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(precision_score(y_test, y_pred, average='macro'))
        performance = (sum(accuracy)/len(accuracy), sum(precision)/len(precision), sum(recall)/len(recall))
    return performance
  
models=['knn','NB','LoG','DT','RF']
for m in models:
  print(classifier(m))
