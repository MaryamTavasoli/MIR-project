import numpy as np
from tqdm import tqdm


class BasicClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        percentage = 0
        if len(sentences) > 0:
            predictions = self.predict(sentences)
            positive_reviews = np.sum(predictions)
            total_reviews = len(sentences)
            percentage = (positive_reviews / total_reviews) * 100.0

        return percentage
