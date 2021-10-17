from typing import Dict
from main import merge_dicts
import pandas as pd

class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """

    def __init__(self, alpha=1):
        self.bag_of_words: Dict = {}
        self.author_words: Dict = {}
        self.all_words: int = 0
        self.unique_words: int = 0
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param X: pd.DataFrame|list - train input/messages
        :param y: pd.DataFrame|list - train output/labels
        :return: None
        """
        for idx in range(len(y)):
            self.bag_of_words[y[idx]] = merge_dicts(self.bag_of_words.get(y[idx], {}), X[idx])
        print("Build bag_of_words")

        for author in self.bag_of_words:
            self.author_words[author] = sum([words for words in self.bag_of_words[author].values()])
            self.all_words += self.author_words[author]
            self.unique_words += len(self.bag_of_words[author])
        print("Fit done")

    def predict_prob(self, message: Dict, label: str):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message
        :param label: str - label
        :return: float - probability P(label|message)
        """
        total_prob = 1

        for word, count in message.items():
            word_count = 0
            for author in self.author_words:
                word_count += self.bag_of_words[author].get(word, 0)
            word_prob = (word_count + self.alpha) / (
                        self.all_words + self.unique_words * self.alpha)

            total_prob *= (((self.bag_of_words[label].get(word, 0) + self.alpha) / (
                    self.author_words[label] + self.unique_words * self.alpha)) ** count) / word_prob
        return total_prob * (self.author_words[label] + self.alpha) / (
                    self.all_words + len(self.author_words) * self.alpha)

    def predict(self, message: Dict):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        maximum_probability = -1
        most_probable_author = ''
        for author in self.bag_of_words:
            probability = self.predict_prob(message, author)
            if probability > maximum_probability:
                maximum_probability = probability
                most_probable_author = author
        return most_probable_author

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param X: pd.DataFrame|list - test data - messages
        :param y: pd.DataFrame|list - test labels
        :return:
        """
        accuracy = 0

        for idx in range(len(y)):
            accuracy += int(self.predict(X[idx]) == y[idx])

        return round(accuracy / len(y) * 100, 2)
