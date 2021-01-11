# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# loading the important datasets
X_test = np.loadtxt('Datasets/testing_features.txt', delimiter=' ')
y_test = np.loadtxt('Datasets/testing_target.txt', delimiter=' ')
# Loding the all the tokens Probabilities files
prob_spam = np.loadtxt('Datasets/probability_spam.txt', delimiter=' ')
prob_ham = np.loadtxt('Datasets/probability_ham.txt', delimiter=' ')
prob_all = np.loadtxt('Datasets/probability_all.txt', delimiter=' ')

# Joint Conditional Probabilities
print('Shape of the dot product is', X_test.dot(prob_spam).shape)

# Probability of spam is 0.311116093
# Taking the log from the each probability
# log(P(tokens | Spam)) - log(P(Tokens)) + log(P(Spam))
# finding the joint probability using log format
# Joint probability for beaing spam
joint_log_spam = X_test.dot(np.log(prob_spam) - np.log(prob_all)) + np.log(0.311116)

# Joint probability for beaing NON spam
joint_log_ham = X_test.dot(np.log(prob_ham) - np.log(prob_all)) + np.log(1 - 0.311116)

"""
Making predictions
We know that spam emails should have the value 1 (True) and ham emails 0 (False) 
"""
prediction = joint_log_spam > joint_log_ham
prediction[10:]*1
y_test[10:]

"""
Model Evaluations
"""

# Accuracy = (Number of correct prediction) / (Total number of predictions)
correct_docs = (y_test == prediction).sum()
print('Docs classified correctly', correct_docs)
numdocs_wrong = X_test.shape[0] - correct_docs
print('Docs classified incorrectly', numdocs_wrong)

accuracy = correct_docs / len(X_test)

# Fraction wrong
fraction_wrong = numdocs_wrong / len(X_test)
print("Fraction classified incorrectly is {:.2%}".format(fraction_wrong))
print('Accuracy of the model is {:.2%}'.format(1 - fraction_wrong))

"""
Visuallising the results
y_axis = 'P(X | Spam)'
x_axis = 'P(X | Nonspam)'
"""
y_axis = 'P(X | Spam)'
x_axis = 'P(X | Nonspam)'
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.scatter(joint_log_ham, joint_log_spam)

# False Positives and False Negatives
np.unique(prediction, return_counts = True)

true_pos = (y_test == 1) & (prediction == 1)
true_pos.sum()

# false positive
false_pos = (y_test == 0) & (prediction == 1)
false_pos.sum()

# False Negative
false_neg = (y_test == 1) & (prediction == 0)
false_neg.sum()

"""
Recall Score = True Positives / (True Positives + False Negatives)
"""
recall_score = true_pos.sum() / (true_pos.sum() + false_neg.sum())
print('Recall Score is {:.2%}'.format(recall_score))

"""
Precision = Ture Positives / (Ture Positives + False Positives) 
"""
precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision score is {:.3}'.format(precision_score))

"""
F1-scor or F-Score = 2 * (Precision * Recall) / (Precision + Recall)
"""
f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
print('F Score is {:.2}'.format(f1_score))