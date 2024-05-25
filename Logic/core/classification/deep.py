import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.classification.basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        train_dataset = ReviewDataSet(x, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                embeddings, labels = batch
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}')

            if self.test_loader:
                eval_loss, _, _, f1_score_macro = self._eval_epoch(self.test_loader, self.model)
                print(
                    f'Epoch [{epoch + 1}/{self.num_epochs}], Validation Loss: {eval_loss:.4f}, F1 Score (Macro): {f1_score_macro:.4f}')
                # Save the best model
                if f1_score_macro > getattr(self, 'best_f1_score', 0):
                    self.best_f1_score = f1_score_macro
                    self.best_model = self.model.state_dict()

        self.model.load_state_dict(self.best_model)
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        test_dataset = ReviewDataSet(x, np.zeros((x.shape[0],)))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        predicted_labels = []
        with torch.no_grad():
            for embeddings, _ in test_loader:
                embeddings = embeddings.to(self.device)
                outputs = self.model(embeddings)
                _, preds = torch.max(outputs, 1)
                predicted_labels.extend(preds.cpu().numpy())

        return predicted_labels

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        total_loss = 0
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings, labels = embeddings.to(self.device), labels.to(self.device)
                outputs = model(embeddings)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        return avg_loss, predicted_labels, true_labels, f1_macro

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        predicted_labels = self.predict(x)
        return classification_report(y, predicted_labels)


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader(
        file_path="/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/classification/preprocessed_reviews_train.csv")
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    deep_classifier = DeepModelClassifier(in_features=100, num_classes=2, batch_size=16, num_epochs=20)
    deep_classifier.set_test_dataloader(X_test, y_test)
    deep_classifier.fit(X_train, y_train)
    report = deep_classifier.prediction_report(X_test, y_test)
    print(report)
    positive_reviews_percentage = deep_classifier.get_percent_of_positive_reviews(X_test)
    print(f"Percentage of positive reviews: {positive_reviews_percentage:.2f}%")
