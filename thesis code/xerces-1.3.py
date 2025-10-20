# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 03:47:25 2019

@author: Smile
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


plt.style.use('ggplot')
df = pd.read_csv('C:/Users/Smile/.spyder-py3/datasets/xerces-1.3.csv')
df.head()
df.isnull().any().sum()> 0
#print(df)
"""******************************CLASS DISTRIBUTION***********************************"""
df = df.sort_values('bug', ascending=False).reset_index(drop=True)
print(df.groupby('bug')['bug'].count())
plt.figure(figsize=(8,6))
sns.countplot(df['bug'])
plt.xticks(fontsize=15)
plt.show()
# Loading the data
train = pd.read_csv('C:/Users/Smile/.spyder-py3/datasets/train/xerces-1.3 Train.csv')
test = pd.read_csv('C:/Users/Smile/.spyder-py3/datasets/test/xerces-1.3 test.csv')
# Store our test passenger IDs for easy access
bug = test['bug']
# Showing overview of the train dataset
print(train.head(3))
original_train = train.copy() # Using 'copy()' allows to clone the dataset, creating a different object with the same values

# Feature engineering steps taken from Sina and Anisotropic, with minor changes to avoid warnings
full_data = [train, test]

drop_elements = ['name','version','tool_name']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
print(train.head(3))

# =============================================================================
# """**********************************CORRELATION COEFFICIENT*********************************"""
# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# =============================================================================
"""***********************************DECISION TREE*********************************"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# seperate the independent and target variable on training data
train_x = train.drop(columns=['bug'],axis=1)
train_y = train['bug']
# seperate the independent and target variable on testing data
test_x = test.drop(columns=['bug'],axis=1)
test_y = test['bug']

model = RandomForestClassifier()

# fit the model with the training data
model.fit(train_x,train_y)

# number of trees used
print('Number of Trees used : ', model.n_estimators)

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('\nTarget on train data',predict_train) 
# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)
print('\nTarget on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(test_y,predict_test)
print('\naccuracy_score on test dataset : ', accuracy_test)

"""***************************ENTROPY OF TRAIN***********************************"""
import numpy as np
from sklearn import metrics 
#from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix 
import pydotplus
def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


    # Model Accuracy, how often is the classifier correct?
print("Accuracy of train 1:",metrics.accuracy_score(train_y,predict_train))
cm_train = confusion_matrix(train_y,predict_train) 
print("Confusion Matrix of train 1: \r\n" , cm_train)

###############################Info Gain ##########################################
"""*********************************Info Gain*****************************************"""

def InfoGain(data,split_attribute_name,target_name="bug"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    print(" Information_Gain:", Information_Gain)
    return Information_Gain


"""**************************************TREE VISUALIZATION************************************"""

import collections
from sklearn import tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred = clf.predict(test_x)


#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  

feature_cols = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]
dot_data = tree.export_graphviz(clf,
                                feature_names=feature_cols,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('xerces-1.3-train1.png')
#print("features ranked",Information_Gain)

"""**********************************INFO GAIN***********************************"""
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred = clf.predict(test_x)

# Model Accuracy, how often is the classifier correct?
#print("Accuracy of train2:",metrics.accuracy_score(train_y,predict_train))

from sklearn.tree import export_graphviz
from IPython.display import Image 
from sklearn.externals.six import StringIO 
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('xerces-1.3-train2.png')
Image(graph.create_png())

"""**************///////////////////////////////////////////////////*******************"""

"""***************************ENTROPY OF TEST***********************************"""

from sklearn import metrics 
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix 
import pydotplus
def entropy1(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


    # Model Accuracy, how often is the classifier correct?
print("Accuracy of test 1:",metrics.accuracy_score(test_y,predict_test))
cm_test = confusion_matrix(test_y,predict_test) 
print("Confusion Matrix of test 1: \r\n" , cm_test)
###############################Info Gain ##########################################
"""**********************Info Gain*****************************************"""
def InfoGain1(data,split_attribute_name,target_name="bug"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    print(" Information_Gain:", Information_Gain)
    return Information_Gain


"""*********************************************************************************"""

import collections
from sklearn import tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred = clf.predict(test_x)


feature_cols = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]
dot_data = tree.export_graphviz(clf,
                                feature_names=feature_cols,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('xerces-1.3-test1.png')
#print("features ranked",Information_Gain)

"""*********************************************************************************"""
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(train_x,train_y)

#Predict the response for test dataset
y_pred = clf.predict(test_x)

from sklearn.tree import export_graphviz
from IPython.display import Image 
from sklearn.externals.six import StringIO 
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('xerces-1.3-test2.png')
Image(graph.create_png())

"""*******************************SVM CLASSIFIER******************************************"""

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(train_x, train_y)

#Predict the response for test dataset
y_pred = knn.predict(test_x)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy of KNearest Neighbour:",metrics.accuracy_score(test_y, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))


"""*********************************Logistic Regression*************************************"""
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(train_x,train_y)

#
y_predict=logreg.predict(test_x)

print("Accuracy of Logistic Regression:",metrics.accuracy_score(test_y, y_predict))
print(confusion_matrix(test_y, y_predict))
print(classification_report(test_y, y_predict))

"""******************************RANDOM FOREST***********************************"""
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_val_score
#Create a Gaussian Classifier
clf1=RandomForestClassifier(n_estimators=1500)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf1.fit(train_x,train_y)

y_predi=clf.predict(test_x)

print("Accuracy of RANDOM FOREST:",metrics.accuracy_score(test_y, y_predi))
print(confusion_matrix(test_y, y_predi))
print(classification_report(test_y, y_predi))
#print( cross_val_score(clf1, train_x, train_y, scoring='accuracy', cv = 10))


"""***************************** SUPPORT VECTOR MACHINE *********************************"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC()
model.fit(train_x,train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)


# predict the target on the test dataset
predict_test = model.predict(test_x)
print("Accuracy of SUPPORT VECTOR MACHINE:",metrics.accuracy_score(test_y, predict_test))
print(confusion_matrix(test_y, predict_test))
print(classification_report(test_y, predict_test))


"""********************************ARTIFICIAL NEURAL NETWORK****************************************"""
# import required modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import numpy as np


def calculate_diagnostic_performance (actual_predicted):
    """ Calculate bug performance.
    
    Takes a Numpy array of 1 and zero, two columns: actual and predicted
    
    Note that some statistics are repeats with different names
    (precision = positive_predictive_value and recall = sensitivity).
    Both names are returned
    
    Returns a dictionary of results:
        
    1) accuracy: proportion of test results that are correct    
    2) sensitivity: proportion of true +ve identified
    3) specificity: proportion of true -ve identified
    4) positive likelihood: increased probability of true +ve if test +ve
    5) negative likelihood: reduced probability of true +ve if test -ve
    6) false positive rate: proportion of false +ves in true -ve patients
    7) false negative rate:  proportion of false -ves in true +ve patients
    8) positive predictive value: chance of true +ve if test +ve
    9) negative predictive value: chance of true -ve if test -ve
    10) precision = positive predictive value 
    11) recall = sensitivity
    12) f1 = (2 * precision * recall) / (precision + recall)
    13) positive rate = rate of true +ve (not strictly a performance measure)
    """
# Calculate results
    actual_positives = actual_predicted[:, 0] == 1
    actual_negatives = actual_predicted[:, 0] == 0
    test_positives = actual_predicted[:, 1] == 1
    test_negatives = actual_predicted[:, 1] == 0
    test_correct = actual_predicted[:, 0] == actual_predicted[:, 1]
    accuracy = np.average(test_correct)
    true_positives = actual_positives & test_positives
    true_negatives = actual_negatives & test_negatives
    sensitivity = np.sum(true_positives) / np.sum(actual_positives)
    specificity = np.sum(true_negatives) / np.sum(actual_negatives)
    positive_likelihood = sensitivity / (1 - specificity)
    negative_likelihood = (1 - sensitivity) / specificity
    false_positive_rate = 1 - specificity
    false_negative_rate = 1 - sensitivity
    positive_predictive_value = np.sum(true_positives) / np.sum(test_positives)
    negative_predictive_value = np.sum(true_negatives) / np.sum(test_negatives)
    precision = positive_predictive_value
    recall = sensitivity
    f1 = (2 * precision * recall) / (precision + recall)
    positive_rate = np.mean(actual_predicted[:,1])
    
    # Add results to dictionary
    performance = {}
    performance['accuracy'] = accuracy
    performance['sensitivity'] = sensitivity
    performance['specificity'] = specificity
    performance['positive_likelihood'] = positive_likelihood
    performance['negative_likelihood'] = negative_likelihood
    performance['false_positive_rate'] = false_positive_rate
    performance['false_negative_rate'] = false_negative_rate
    performance['positive_predictive_value'] = positive_predictive_value
    performance['negative_predictive_value'] = negative_predictive_value
    performance['precision'] = precision
    performance['recall'] = recall
    performance['f1'] = f1
    performance['positive_rate'] = positive_rate

    return performance



def normalise (X_train,X_test):
    """Normalise X data, so that training set has mean of zero and standard
    deviation of one"""
    
    # Initialise a new scaling object for normalising input data
    sc=StandardScaler() 
    # Set up the scaler just on the training set
    sc.fit(X_train)
    # Apply the scaler to the training and test sets
    X_train_std=sc.transform(X_train)
    X_test_std=sc.transform(X_test)
    return X_train_std, X_test_std


def print_bug_results (performance):
    """Iterate through, and print, the performance metrics dictionary"""
    
    print('\n ANN Model performance measures:')
    print('-------------------------------------------------')
    for key, value in performance.items():
        print (key,'= %0.3f' %value) # print 3 decimal places
    return

def split_data (data_set, split=0.25):
    """Extract X and y data from data_set object, and split into tarining and
    test data. Split defaults to 75% training, 25% test if not other value 
    passed to function"""
    
    X=data_set.iloc[:, :-1].values
    y=data_set.iloc[:, 1].values
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=split, random_state=0)
    return X_train,X_test,y_train,y_test

def test_model(model, X, y):
    """Return predicted y given X (attributes)"""
    
    y_pred = model.predict(X)
    test_results = np.vstack((y, y_pred)).T
    return test_results

def train_model (X, y):
    """Train the model """
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-8, hidden_layer_sizes=(50, 5),
                        max_iter=100000, shuffle=True, learning_rate_init=0.001,
                        activation='relu', learning_rate='constant', tol=1e-7,
                        random_state=0)
    model.fit(X_train_std, y_train)   
    return model

###### Main code #######

# Load data
data_set = pd.read_csv('C:/Users/Smile/.spyder-py3/datasets/train/xerces-1.3 Train.csv')
drop_elements = ['name','version','tool_name']
data_set = data_set.drop(drop_elements, axis = 1)
# Split data into trainign and test sets
X_train,X_test,y_train,y_test = split_data(data_set, 0.25)

# Normalise data
X_train_std, X_test_std = normalise(X_train,X_test)

# Train model
model = train_model(X_train_std,y_train)

# Produce results for test set
test_results = test_model(model, X_test_std, y_test)

# Measure performance of test set predictions
performance = calculate_diagnostic_performance(test_results)

# Print performance metrics
print_bug_results(performance)

#cross_validation
print( "Cross Validation : ",cross_val_score(clf1, train_x, train_y, scoring='accuracy', cv = 10))

"""**********************************CORRELATION COEFFICIENT*********************************"""
# =============================================================================
# colormap = plt.cm.viridis
# plt.figure(figsize=(12,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# =============================================================================

# =============================================================================
# # Import the model we are using
# from sklearn.ensemble import RandomForestRegressor
# # Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# # Train the model on training data
# rf.fit(train_x, train_y);
# # Use the forest's predict method on the test data
# predictions = rf.predict(test_x)
# # Calculate the absolute errors
# errors = abs(predictions - test_y)
# # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# accuracy_score(test_y, predictions.round(), normalize=False)
# print("Accuracy of SUPPORT VECTOR MACHINE:",metrics.accuracy_score(test_y, predict_test))
# 
# #print(classification_report(test_y, predictions))
# =============================================================================
# =============================================================================
# import pandas as pd
# # =============================================================================
# # dataset=pd.read_csv('C:/Users/Smile/.spyder-py3/datasets/xerces-1.3.csv')
# # drop_elements = ['name','version','tool_name']
# # dataset = dataset.drop(drop_elements, axis = 1)
# # =============================================================================
# from sklearn.ensemble import RandomForestRegressor
# # Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# # Train the model on training data
# rf.fit(train_x, train_y);
# 
# predictions = rf.predict(test_x)
# =============================================================================
