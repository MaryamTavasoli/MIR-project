import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader
import tqdm
import pandas as pd


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        # print(x)
        # print(y)
        self.classes, counts = np.unique(y, return_counts=True)
        self.num_classes = len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape
        self.prior = counts / self.number_of_samples

        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for idx, cls in enumerate(self.classes):
            li = []
            for i in range(len(y)):
                if y[i] == cls:
                    li.append(i)
            x_class = x[li]
            print(self.number_of_features)
            for j in range(self.number_of_features):
                self.feature_probabilities[idx, j] = (x_class[:, j].sum() + self.alpha) / (
                        x_class.sum() + self.alpha * self.number_of_features)
                # print(self.feature_probabilities[idx, j])
                # print(j)
        self.log_probs = np.log(self.feature_probabilities)

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        log_prior = np.log(self.prior)
        log_likelihood = x @ self.log_probs.T
        log_posterior = log_prior + log_likelihood
        return self.classes[np.argmax(log_posterior, axis=1)]

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        predictions = self.predict(x)
        return classification_report(y, predictions)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        embeddings = self.cv.transform(sentences)
        predictions = self.predict(embeddings.toarray())
        positive_reviews = np.sum(predictions == 'positive')
        return (positive_reviews / len(predictions)) * 100


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    review_loader = ReviewLoader(
        file_path="/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/classification/preprocessed_reviews_train.csv")
    review_loader.load_data()
    reviews = review_loader.review_tokens
    review_strings = []
    for review in reviews:
        review_string = ""
        for word in review:
            review_string += word
            review_string += " "
        review_strings.append(review_string)
    labels = review_loader.sentiments
    X_train, X_test, y_train, y_test = train_test_split(review_strings, labels, test_size=0.2, random_state=42)
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    naive_bayes_classifier = NaiveBayes(count_vectorizer)
    naive_bayes_classifier.fit(X_train_counts, y_train)
    predictions = naive_bayes_classifier.predict(X_test_counts)
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)
    positive_reviews_percentage = naive_bayes_classifier.get_percent_of_positive_reviews(X_test)
    print(f"Percentage of positive reviews: {positive_reviews_percentage:.2f}%")
    # df1 = pd.read_csv("/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/classification/preprocessed_crawled_reviews.csv")
    # crawled_reviews = df1['review'].astype(str).tolist()
    # review_crawled_strings = []
    # for review in crawled_reviews:
    #     review_crawled_string = ""
    #     for word in review[0]:
    #         review_crawled_string += word
    #         review_crawled_string += " "
    #     review_crawled_strings.append(review_crawled_string)
    # count_vectorizer = CountVectorizer()
    # X_crawled = count_vectorizer.fit_transform(review_crawled_strings)
    # crawled_predictions = naive_bayes_classifier.predict(X_crawled)
    # print(crawled_predictions)
    # df1['NaiveBayes_prediction'] = crawled_predictions
