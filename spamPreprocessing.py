# importing libraries
from os import walk
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split

# Readings files with path
SPAM_1_PATH = 'Datasets/spam_assassin_corpus/spam_1'
SPAM_2_PATH = 'Datasets/spam_assassin_corpus/spam_2'
EASY_NONSPAM_1_PATH = 'Datasets/spam_assassin_corpus/easy_ham_1'
EASY_NONSPAM_2_PATH = 'Datasets/spam_assassin_corpus/easy_ham_2'

# Email Body Extraction using Generator Function
def email_body_generator(path):
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            filepath = join(root, file_name)
            stream = open(filepath, encoding='latin-1')
            is_body = False
            lines = []
            
            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True
            
            stream.close()
            
            email_body = '\n'.join(lines)
            
            yield file_name, email_body

# Calling the Generator Function
def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index=row_names)

# Creating the DataFrame from the df_from_directory
spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, 1))

first_mail = spam_emails.iloc[1, 0]
spam_emails.shape

non_spam_emails = df_from_directory(EASY_NONSPAM_1_PATH, 0)
non_spam_emails = non_spam_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, 0))
non_spam_emails.shape

# Whole Datasets including Spam and Non Spam Emails
emails_dataset = pd.concat([spam_emails, non_spam_emails])
print('Shapes of the entire dataframe', emails_dataset.shape)
mail_message = emails_dataset.iloc[4, 0]

"""
Data Cleaning
    Checking the Missing Values
"""
# checking if any message bodies are null
emails_dataset['MESSAGE'].isnull().values.any()

# Checking if there are empty emails (string length zero)
(emails_dataset['MESSAGE'].str.len() == 0).any()

# How many are empty emails (string length zero)
(emails_dataset['MESSAGE'].str.len() == 0).any().sum()

# Locating the empty emails
emails_dataset[emails_dataset['MESSAGE'].str.len() == 0].index

# Removing the System entries from the dataset
emails_dataset.drop(['cmds'], inplace = True)

# Adding the Document IDs to Track emails
documnet_ids = range(0, len(emails_dataset.index))
emails_dataset['DOC_ID'] = documnet_ids

emails_dataset['FILE_NAME'] = emails_dataset.index
emails_dataset.set_index('DOC_ID', inplace = True)

"""
Saving the files in to different file format
"""
# Into JSON Format
emails_dataset.to_json('Datasets/emails_json_dataset.json')
emails_dataset.to_csv('Datasets/emails_csv_dataset.csv')

"""
Visualising the Collected Information
"""
spam_amount = emails_dataset.CATEGORY.value_counts()[1]
ham_amount = emails_dataset.CATEGORY.value_counts()[0]
custom_colors = ['#ff7675', '#74b9ff']

category_names = ['Spam', 'Lagit Mail']
sizes = [spam_amount, ham_amount]
plt.pie(sizes, labels=category_names, colors=custom_colors, startangle=90, autopct='%1.0f%%', explode=[0, 0.1])

"""
NLP Text Preprocessing
"""
email_body = emails_dataset['MESSAGE'][0]
# Function for dealing with the cleaning text
def clean_message(message, stemmer=PorterStemmer(), stop_words=set(stopwords.words('english'))):
    # converting the string into the lowercase and splits up the words
    words = word_tokenize(message.lower())
    
    filtered_words = []
    for word in words:
        # Removing the stopwords and punctuations
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    return filtered_words

clean_message(email_body)

def clean_msg_no_html(message, stemmer=PorterStemmer(), stop_words=set(stopwords.words('english'))):
    # Removing the HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    # converting the string into the lowercase and splits up the words
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    for word in words:
        # Removing the stopwords and punctuations
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    return filtered_words

clean_msg_no_html(emails_dataset.at[3, 'MESSAGE'])

# Using the apply() on all the messages in the dataFrame
nested_list = emails_dataset.MESSAGE.apply(clean_msg_no_html)

"""
Generating Vocablary 
"""
stemmed_list = [item for sublist in nested_list for item in sublist]

# Unique words 
unique_words = pd.Series(stemmed_list).value_counts()
print('Number of Unique words', unique_words.shape[0])

# Frequent 2500 words
frequent_words = unique_words[0:2500]
print('Most common words: \n', frequent_words[:10])

# Creating the Vocabulary DataFrame with a word_id
word_ids = list(range(0, 2500))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'

# Saving the vocabulary as a CSV file
vocab.to_csv('Datasets/emails_vocabulary.csv', index_label = vocab.index.name, header = vocab.VOCAB_WORD.name)

"""
Generating the Sparse Matrix and fratures
"""
# Creating the dataFrame with one word per column
type(nested_list)
type(nested_list.tolist())
words_columns_df = pd.DataFrame.from_records(nested_list.tolist())

# Spliting the data into train and Test Datasets
X_train, X_test, y_train, y_test = train_test_split(words_columns_df, emails_dataset.CATEGORY, test_size = 0.3, random_state = 42)

X_train.index.name = X_test.index.name = 'DOC_ID'

# Creating the word Index
word_index = pd.Index(vocab.VOCAB_WORD)

# Creating A Sparse Matrix for the Training Data set
def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe
    def : A dataframe with words in the columns with a documnet id as an index (X_train, X_test) 
    indexed_words : index of words ordered by  word_id
    labels : category as a series (y_train, y_test)
    """
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                
                item = {'LABEL' : category, 'DOC_ID' : doc_id, 'OCCURENCE' : 1, 'WORD_ID' : word_id}
                
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)

# Calling make_sparse_matrix() for training datasets 
########### It will take lots of time to execute it
sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)

# Combine Occurrences with the pandas groupby() method  for training datasets 
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()

# Saving the Training data as .txt file  for training datasets 
np.savetxt('Datasets/train_data.txt', train_grouped, fmt = '%d')

# Calling make_sparse_matrix() for training datasets 
########### It will take lots of time to execute it
sparse_test_df = make_sparse_matrix(X_test, word_index, y_test)

# Combine Occurrences with the pandas groupby() method  for training datasets 
test_grouped = sparse_test_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()

# Saving the Training data as .txt file  for training datasets 
np.savetxt('Datasets/test_data.txt', test_grouped, fmt = '%d')


