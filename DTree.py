#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:58:54 2019

@author: niloofarzarifi
"""
#Decision Tree, cross validation and ensemble models
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import os

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from random import randint
import sklearn
print (sklearn.__version__) #0.19.1
os.getcwd()
os.chdir('/Users/niloofarzarifi/Desktop/Udacity/khaneh/Watson-IBM-customer/')
data = pd.read_csv('/Users/niloofarzarifi/Desktop/Udacity/khaneh/Watson-IBM-customer/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')
data.head()


import seaborn as sns

fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.savefig('heatmap',dpi=600,bbox_inches='tight')


#look into individual vars 
plt.scatter(data["Total Claim Amount"],data['Monthly Premium Auto'])
##or
sns.regplot(x='Total Claim Amount',y='Monthly Premium Auto',data=data) #shows regression
sns.regplot(x='Customer Lifetime Value',y='Monthly Premium Auto',data=data)
sns.regplot(x='Total Claim Amount',y='Income',data=data)
plt.savefig('foo',dpi=600,bbox_inches='tight')




##--------------------
????????????????


from pandas.plotting import scatter_matrix
scatter_matrix(data,alpha=0.2, figsize=(6,6), diagonal='kde')
#Change label rotation
[s.xaxis.label.set_rotation(90) for s in sm.reshape(-1)]
[s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
#May need to offset label when rotating to prevent overlap of figure
[s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]

#Hide all ticks
[s.set_xticks(()) for s in sm.reshape(-1)]
[s.set_yticks(()) for s in sm.reshape(-1)]
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()

#------------------------------------------------------------
#first try for decision tree

data.columns.tolist()
y = data['Response']   # select target,
X=data.select_dtypes(include=['int64', 'float64'])
filter_col=list(X.columns)
print ('int and float columns:', len(list(X.columns)))
print ('all columns:', len(list(data.columns.values)))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_test)
dtree.tree_.impurity #keeps the gini values of the tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):

    #print (filter_col[indices[f]])
    print("%d. feature %d (%f) name: %s " % (f + 1, indices[f], importances[indices[f]], filter_col[indices[f]] ))


###############################
    #must import the following lines
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
# Create DOT data
dot_data = tree.export_graphviz(dtree, out_file=None, 
                                feature_names=filter_col,  
                                class_names=['Yes','No'])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())

# Create PDF
graph.write_pdf("FirstDT.pdf")

# Create PNG
graph.write_png("FirstDT.png")
# the tree need pruning
###############################################################
##ROC plot
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_probs = dtree.predict_proba(X_test)
probability = dtree.predict_proba(X_test)[:,1] # only second col

y_test = y_test.apply(lambda z : 1 if z=='Yes' else 0) #convert binary to 0 and 1

fpr, tpr, threshold = roc_curve(y_test,probability)
plt.plot(fpr,tpr)
plt.xlabel("FPR")
plt.ylabel('TPR')
plt.title('ROC with AUC score: {}'.format(roc_auc_score(y_test,y_pred_probs)))

plt.show()
roc_auc_score(y_test,probability) #0.97 sounds really good!
auc(fpr, tpr)


from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Evaluate predictions
print ('Accuracy: %.5f' % accuracy_score(y_test, probability))
print ('F1 score: %.5f' % f1_score(y_test, probability))
print ('AUROC: %.5f' % roc_auc_score(y_test, probability))
print ('AUPRC: %.5f' % average_precision_score(y_test, probability))


# To plot ROC and PRC
from sklearn.metrics import roc_curve, precision_recall_curve

# Compute FPR, TPR, Precision by iterating classification thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:, 1])
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_probs[:, 1])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(8,5))
axes[0].plot(fpr, tpr)
axes[0].set_xlabel('FPR')
axes[0].set_ylabel('TPR')
axes[0].set_title('ROC')

axes[1].plot(recall, precision)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PRC')
plt.savefig('foo',dpi=600,bbox_inches='tight')

#-----------------------------------------------------
#before cross validation let's try few tests
y = data['Response']   # select target,
X=data.select_dtypes(include=['int64', 'float64'])
filter_col=list(X.columns)
#for ROC we need integer target
y = y.apply(lambda z : 1 if z=='Yes' else 0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
train_results = []
test_results = []

for i in range(3,20):
   dt = DecisionTreeClassifier(criterion='gini',max_depth=i)
   dt.fit(X_train,y_train)
   train_pred = dt.predict(X_train)
   fpr, tpr, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(fpr, tpr)
   # Add auc score to previous train results
   train_results.append(roc_auc)
   y_pred = dt.predict(X_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
Line1,=plt.plot(range(3,20), train_results,color='blue',label="Train AUC")
Line2,=plt.plot(range(3,20), test_results, color='red',label="Test AUC")

plt.legend(handler_map={Line1: HandlerLine2D(numpoints=2)})
plt.xlabel("Tree depth")
plt.ylabel('AUC score')
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()

#our model shows there is no overfits for large depth values. The tree somehow predicts all 
#of the train data and it does not fails to generalize the findings for new data
#we can try the same loop for min_samples_splits,min_samples_leafs, max_features
##https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3
#The link above, explains overfitting
#----------------------------------------------------


#StratifiedKFold
##########################
#fresh start
y = data['Response']   # select target,
X=data.select_dtypes(include=['int64', 'float64'])
filter_col=list(X.columns)
from sklearn import tree
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from pprint import pprint
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split

#X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=42)

print "Optimized tree depth using CV and StratifiedKFold "
cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
depth = []
for i in range(3,10):
    dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)

    # Perform 7-fold cross validation
    scores = cross_val_score(estimator=dtree, X=X, y=y, cv = cv, n_jobs=None)
    depth.append((i,scores.mean()))
print(depth)  #score dropped down from 0.96 to 0.86.
type(depth)
depth[0][0]
depth[0][1]
newdepth=pd.DataFrame(depth) 

plt.scatter(newdepth[0],newdepth[1])
plt.xlabel("max_depth")
plt.ylabel("accuracy_score")
plt.savefig('foo',dpi=600,bbox_inches='tight')

#how about ROC? probabely the accuracy is around 0.6 again
#####################







##############################################################################
##here I fixed max_depth=5 and did 10 kfold  ?????kfold and startified????
#StratifiedKFold::this cross-validation object is a variation of KFold that returns stratified folds. 
##The folds are made by preserving the percentage of samples for each class.
#Kfold::Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds 
#(without shuffling by default).
#GroupKFold??
#RepeatedKFold??
#RepeatedStratifiedKFold??
#cross_val_score:it is Kfold but already has the loop inside
#GridSearchCV::Exhaustive search over specified parameter values for an estimator.
#Lasso
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
scores=[] ;max_score=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    dtree=DecisionTreeClassifier(criterion='gini',max_depth=5)
    #train the model
    dtree.fit(X_train,y_train)
    #see performance score
    current_score = dtree.score(X_test,y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score

import statistics 
print ('all scores: {}'.format(scores))
print ('mean score: ', statistics.mean(scores))








#-------------------------------------------------------------------------------
###RandomForestClassifier
#test_size=40%

#??????????????NameError: name 'model_selection' is not defined
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
#two lines above problematic!! only import model_selection
from sklearn import model_selection

#
#Kfold no need to split dataset
seed = 7
max_features = len(filter_col)  #choose number of features
kfold = model_selection.KFold(n_splits=10, random_state=seed)

#optimize number of trees
for num_trees in range(3,20):
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, y, cv=kfold)
    print("RandomForestClassifier, Cross_val_score=",results.mean() )
    

#
#results sounds great but if I add max_depth=5 I have many missclassified points


#How to Visualize a Decision Tree from a Random Forest in Python using Scikit-Learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y ,random_state=0)    
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
model = RandomForestClassifier(n_estimators=10,max_depth=5)
pprint(model.get_params())
# Train
model.fit(X_train, y_train)
# Extract single tree
estimator = model.estimators_[3] #we can change the tree number:)

# Create DOT data
dot_data = tree.export_graphviz(estimator, out_file=None, 
                                feature_names=filter_col,  
                                class_names=['Yes','No'],
                                rounded = True, proportion = False,
                                precision = 2, filled = True)
                                

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph
Image(graph.create_png())

# Create PDF
graph.write_pdf("RF-3th.pdf")

#The gini score is a metric that quantifies the purity of the node/leaf 
#dtree.tree_.impurity is the gini array but it doesnt support RF
#----------------------------------------------------
# good interpratable example of GridSearchCV  RandomizedSearchCV
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import cross_val_score
from time import time
from operator import itemgetter
from scipy.stats import randint

# This function takes the output from the grid or random search,
# prints a report of the top models and returns the best parameter setting.
def report(grid_scores, n_top=3):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters



# The param_grid is the set of parameters that will be
# tested– be careful not to list too many options, because all combinations will be tested!
def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                                         len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return top_params

## define target and features and split X_train and X_test
y = data['Response']   # select target,
X=data.select_dtypes(include=['int64', 'float64'])
filter_col=list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# split the data into 10 parts
# fit on 9-parts
# test accuracy on the remaining part
print("-- 10-fold cross-validation "
      "[using setup without GridSearchCV]")

dt_old = DecisionTreeClassifier(min_samples_split=20,
                                random_state=0)
dt_old.fit(X, y)
scores = cross_val_score(dt_old, X_train, y_train, cv=10)
from __future__ import print_function
print("mean: {:.3f} (std: {:.3f})".format(scores.mean() ,
      scores.std()) ,
      end='\n\n')


print("-- Grid Parameter Search (GridSearchCV) via 10-fold CV")

# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [100, 200, 300],
              "max_depth": [3, 5, 6],
              "min_samples_leaf": [50, 100, 150],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

dt = DecisionTreeClassifier()
ts_gs = run_gridsearch(X_train, y_train, dt, param_grid, cv=10)


# test the retuned best parameters
# replicate the cross-validation results:
print("\n\n-- Testing best parameters [Grid]...")
dt_ts_gs = DecisionTreeClassifier(**ts_gs)
scores = cross_val_score(dt_ts_gs, X_test, y_test, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                          scores.std()),
      end="\n\n")

dt_ts_gs.fit(X_train, y_train)  #could be final optimized model based on defined param_grid

################################################################
#----------------------------------------------------------------------------------------------
##Voting Ensemble for Classification
from sklearn.ensemble import ExtraTreesClassifier ####ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier ##bagged decision tree
from sklearn.ensemble import AdaBoostClassifier  ##AdaBoostClassifier 
from sklearn.ensemble import GradientBoostingClassifier  ##GradientBoostingClassifier

from sklearn import model_selection

kfold = model_selection.KFold(n_splits=5, random_state=0)

model1 = ExtraTreesClassifier(n_estimators=5, max_features=len(filter_col),max_depth=5) #n_estimators number of trees
model2 = RandomForestClassifier(n_estimators=5, max_features=len(filter_col),max_depth=5)
cart = DecisionTreeClassifier()
model3 = BaggingClassifier(base_estimator=cart, n_estimators=5, random_state=0,max_depth=5)
model4 = AdaBoostClassifier(n_estimators=5, random_state=0)
model5 = GradientBoostingClassifier(n_estimators=5, random_state=0,max_depth=5)
#A Voting classifier model combines multiple different models 
#(i.e., sub-estimators) into a single model, which is (ideally) stronger than 
#any of the individual models alone.



from sklearn.ensemble import VotingClassifier

estimators =[('ExtraTree',model1),('RF',model2),('Bagging',model3),
             ('AdaBoost',model4),('GBoost',model5)]

ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())

from sklearn.metrics import classification_report, confusion_matrix
ensemble.fit(X,y)
y_pred=ensemble.predict(X_test)
#poor result here due to the fix max_depth
print('confusion_matrix\n',
       confusion_matrix(y_test,y_pred))

#----------------------------------------------------------------------------------------










