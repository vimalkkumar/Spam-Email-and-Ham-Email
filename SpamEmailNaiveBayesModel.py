# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

# Loading the dataset
emails = pd.read_csv('Datasets/emails_csv_dataset.csv') 

# Sorting the data-points according to index value
emails.sort_index(inplace = True)

# Spliting the data-points into features and target
features = emails.iloc[:, 1]
target = emails.iloc[:, 2]

# Features Extractions of the dataset
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = 'english')
features = vectorizer.fit_transform(features)

# Vocabulary from the dataset
vocabulary_list = vectorizer.vocabulary_

# Spliting the dataset in training and testing 
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 0)

# Its time to implement the Naive Bayes Algorithm
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(features_train, target_train)

# Its time to predict the result
target_predict = classifier.predict(features_test)

# Its time to check the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test, target_predict)

# plotting the confusion_matrix
sbn.heatmap(cm, annot = True, fmt = '.5g')
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.show()

# Total number of correct prediction
print('Total number of correct predictions {}'.format(cm[0:1, 0]))

# Total number of incorrect prediction
print('Total number of incorrect predictions {}'.format(cm[1:, 0]))

# recall score, precision score and F1 Score of the result
from sklearn.metrics import recall_score, precision_score, f1_score
print('Recall score {:.2%}'.format(recall_score(target_test, target_predict)))
print('Precision score {:.2%}'.format(precision_score(target_test, target_predict)))
print('F1 score {:.2%}'.format(f1_score(target_test, target_predict)))

# Accuracy of the model
from sklearn.metrics import accuracy_score
print('Accuracy of the model {:.2%}'.format(accuracy_score(target_test, target_predict)))

"""
Examplary Testing
Spam text : 
    1. Quick Loans
    A [redacted] loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply stop
    2. Debt forgiveness
    Due to a new legislation, those struggling with debt can now apply to have it written off. For more information text the word INFO or to opt out text STOP
    3.  Pension
    Our records indicate your Pension is under performing to see higher growth and up to 25% cash release reply PENSION for a free review. To opt out reply STOP'
NonSpam text :
    1. Your Security Rules allow anyone on the internet to read or write to your database. Without strong Security Rules your data is vulnerable to attackers stealing, modifying, or deleting data as well as performing costly operations.
    2. Hi, Mohan. Forward this email to people you want to join ABCD. They can use this link to join your org. Happy sharing!
    3. We do see potential in this partnership. However, considering the stage of the company and skills we can leverage immediately, we need to buy some time before we can come back to you to make it a meaningful association.
"""

example = {
    'A [redacted] loan for £950 is approved for you if you receive this SMS. 1 min verification & cash in 1 hr at www.[redacted].co.uk to opt out reply stop',
    'Due to a new legislation, those struggling with debt can now apply to have it written off. For more information text the word INFO or to opt out text STOP',
    'Our records indicate your Pension is under performing to see higher growth and up to 25% cash release reply PENSION for a free review. To opt out reply STOP',
    'Your Security Rules allow anyone on the internet to read or write to your database. Without strong Security Rules your data is vulnerable to attackers stealing, modifying, or deleting data as well as performing costly operations.',
    'Hi, Mohan. Forward this email to people you want to join ABCD. They can use this link to join your org. Happy sharing!',
    'We do see potential in this partnership. However, considering the stage of the company and skills we can leverage immediately, we need to buy some time before we can come back to you to make it a meaningful association.'
     }

# vactorizering the examples
vectorizer_example = vectorizer.transform(example)

# predicting the result
classifier.predict(vectorizer_example)

