import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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
        y_pred = self.predict(x)
        return classification_report(y, y_pred)


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader(file_path="/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/classification/preprocessed_reviews_train.csv")
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    svm_classifier = SVMClassifier()
    svm_classifier.fit(X_train, y_train)
    report = svm_classifier.prediction_report(X_test, y_test)
    print(report)
    positive_reviews_percentage = svm_classifier.get_percent_of_positive_reviews(X_test)
    print(f"Percentage of positive reviews: {positive_reviews_percentage:.2f}%")
