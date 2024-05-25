import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Logic.core.word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = FastText(preprocessor=None, method='skipgram')
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df = pd.read_csv(self.file_path)
        reviews = df['review'].astype(str).tolist()
        sentiments = df['sentiment'].tolist()
        self.review_tokens = [review.split() for review in tqdm.tqdm(reviews, desc="Processing reviews")]
        self.sentiments = sentiments
        self.fasttext_model.load_model(path="/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core"
                                            "/word_embedding/FastText_model.bin")

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        review_embeddings = []
        for review in tqdm.tqdm(self.review_tokens, desc="Getting embeddings"):
            token_embeddings = []
            for token in review:
                embedding = self.fasttext_model.get_query_embedding(token, do_preprocess=False)
                token_embeddings.append(embedding)
            review_embedding = np.mean(token_embeddings, axis=0)
            review_embeddings.append(review_embedding)

        self.embeddings = np.array(review_embeddings)

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        label_encoder = LabelEncoder()
        encoded_sentiments = label_encoder.fit_transform(self.sentiments)
        print(self.embeddings)
        x_train, x_test, y_train, y_test = (
            train_test_split(self.embeddings, encoded_sentiments, test_size=test_data_ratio, random_state=42))
        return x_train, x_test, y_train, y_test
