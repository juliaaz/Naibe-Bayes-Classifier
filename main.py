"""
Module for implementation Naive Bayes Classifier.
"""
import string
from collections import Counter
from typing import Dict
from bayesian_classifier import *
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    data = pd.read_csv(data_file)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.drop('id', 1)

    X = []
    y = data['author']
    banwords = stopwords.words('english')
    for index, row in data.iterrows():
        row['text'] = row['text'].translate(str.maketrans('', '', string.punctuation))
        X.append(dict(Counter([word.lower() for word in word_tokenize(row['text']) if word not in banwords])))
    return X, y


def merge_dicts(dict1, dict2):
    """
    Merges all the dictionaries, so in result bag of words can be created.
    """
    if len(dict1) < len(dict2):
        dict1, dict2 = dict2, dict1

    for key, value in dict2.items():
        dict1[key] = dict1.get(key, 0) + value
    return dict1



if __name__ == '__main__':
    train_X, train_y = process_data("data/train.csv")
    print("Train parse done")
    test_X, test_y = process_data("data/test.csv")
    print("Test parse done")

    classifier = BayesianClassifier()
    classifier.fit(train_X, train_y)
    print(f"Model score: {classifier.score(test_X, test_y)}")
