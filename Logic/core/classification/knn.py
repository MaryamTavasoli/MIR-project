import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.x_train = x
        self.y_train = y
        return self

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
        predictions = []
        for x_test in tqdm(x):
            distances = np.linalg.norm(self.x_train - x_test, axis=1)
            # print(distances)
            k_nearest_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_label = unique_labels[counts.argmax()]
            predictions.append(most_common_label)
        return np.array(predictions)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader(
        file_path="/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/classification/preprocessed_reviews_train.csv")
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    knn_classifier = KnnClassifier(10)
    knn_classifier.fit(X_train, y_train)
    report = knn_classifier.prediction_report(X_test, y_test)
    print(report)
    positive_reviews_percentage = knn_classifier.get_percent_of_positive_reviews(X_test)
    print(f"Percentage of positive reviews: {positive_reviews_percentage:.2f}%")
