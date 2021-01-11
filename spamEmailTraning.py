import numpy as np
import pandas as pd

# Reading files from the folder
train_data = np.loadtxt('Datasets/train_data.txt', delimiter = ' ', dtype = int)
test_data = np.loadtxt('Datasets/test_data.txt', delimiter = ' ', dtype = int)

print('Number of rows in the Training dataset', train_data.shape[0])
print('Number of rows in the Testing dataset', test_data.shape[0])

# How to create an Empty DataFrame
columns_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, 2500))
columns_names[:5]

index_names = np.unique(train_data[:, 0])

full_train_data = pd.DataFrame(index = index_names, columns = columns_names)
full_train_data.fillna(value = 0, inplace = True)

# Creating the full matrix from the sparse Matrix
def make_full_matrix(sparse_matrix, nr_words, doc_idx = 0, word_idx = 1, cat_idx = 2, freq_idx = 3):
    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe.
    keyword arguments:
        sparse_matrix - numpy array
        nr-words -- size of the vocabulary. Total number of tokens
        doc_idx -- position of the documnet id in the sparse matrix. default: 1st column
        word_idx -- position of the word id in the sparse matrix. drfault 2nd column
        cat_idx -- position of the label (spam is 1, nonspam is 0). default: 3rd column
        freq_idx -- position of the occurrence of the word in sparse matrix. default: 4th column
    """
    columns_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, 2500))
    doc_id_names = np.unique(train_data[:, 0])
    full_train_data = pd.DataFrame(index = index_names, columns = columns_names)
    full_train_data.fillna(value = 0, inplace = True)
    
    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        
        full_train_data.at[doc_nr, 'DOC_ID'] = doc_nr
        full_train_data.at[doc_nr, 'CATEGORY'] = label
        full_train_data.at[doc_nr, word_id] = occurrence
    
    full_train_data.set_index('DOC_ID', inplace = True)
    return full_train_data

full_train_data_matrix = make_full_matrix(train_data, 2500)

"""
Training the Naive Bayes Model
"""
# probability of Spam is
prob_spam = full_train_data_matrix.CATEGORY.sum()/ full_train_data_matrix.CATEGORY.size
print('Probability of Spam is', prob_spam) 

# Total Numner of Words / Tokens
full_train_features = full_train_data_matrix.loc[:, full_train_data_matrix.columns != 'CATEGORY']
full_train_features.head()

# Number of Tokens in spam and Ham Emails
email_lengths = full_train_features.sum(axis = 1)
spam_lengths = email_lengths[full_train_data_matrix.CATEGORY == 1]
spam_lengths.shape

# Spam Word counts
spam_wc = spam_lengths.sum()

# Non Spam emails lengths
ham_lengths = email_lengths[full_train_data_matrix.CATEGORY == 0]
ham_lengths.shape

# Non spam word count
nonspam_wc = ham_lengths.sum()

# Total word counts
total_wc = email_lengths.sum()
total_wc.shape
email_lengths.shape[0] - spam_lengths.shape[0] - ham_lengths.shape[0]

# Figuring out if spam emails or non spam emails tend to be longer? Which category contains more tokens?
print('Average number of words in spam emails {:.0f}', format(spam_wc / spam_lengths.shape[0]))
print('Average  number of words in ham emails {:.3f}'. format(nonspam_wc / ham_lengths.shape[0]))

"""
Summing the tokens occuring in spam
"""
full_train_features.shape
train_spam_tokens = full_train_features.loc[full_train_data_matrix.CATEGORY == 1]

summed_spam_tokens = train_spam_tokens.sum(axis = 0) + 1

# Summing the tokens Occuring in Ham
train_ham_tokens = full_train_features.loc[full_train_data_matrix.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens[2499].sum() + 1

# P(Token | Spam) - Probability that a token occurs given the email is spam
prob_tokens_spam = summed_spam_tokens / (spam_wc + 2500)
prob_tokens_spam.sum()

# P(Token | ham) - Probability that a token occurs given the email is NOn Spam
prob_tokens_nonspam = summed_ham_tokens / (nonspam_wc + 2500)
prob_tokens_nonspam.sum()

# P(tokens) - Probability that token Occurs
prob_tokens_all = full_train_features.sum(axis = 0) / total_wc
prob_tokens_all.sum()

# Saving the Probability files
np.savetxt('Datasets/probability_spam.txt', prob_tokens_spam)
np.savetxt('Datasets/probability_ham.txt', prob_tokens_nonspam)
np.savetxt('Datasets/probability_all.txt', prob_tokens_all)

