import pandas as pd
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        pass

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        ids = []
        titles = []
        synopses = []
        summaries = []
        reviews = []
        genres = []

        # Extract information for each movie
        for doc_id, info in data['documents'].items():
            ids.append(doc_id)
            titles.append(info['title'])
            if 'synopses' in info:
                synopses.append(info['synopsis'])
            else:
                synopses.append([])
            if 'summaries' in info:
                summaries.append(info['summaries'])
            else:
                summaries.append([])
            # Assuming 'reviews' key exists and contains a list of reviews
            if 'reviews' in info:
                reviews.append(info['reviews'])
            else:
                reviews.append([])
            genres.append(info['geners'])

        # Create a DataFrame
        df = pd.DataFrame({
            'ID': ids,
            'Title': titles,
            'Synopses': synopses,
            'Summary': summaries,
            'Reviews': reviews,
            'Genres': genres
        })

        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        # preprocessor1 = Preprocessor(df['Title'].tolist())
        # df['Title'] = preprocessor1.preprocess()
        #
        # # Pre-process text data independently
        # preprocessor2 = Preprocessor(df['Synopsis'].tolist())
        # df['Synopsis'] = preprocessor2.preprocess()
        #
        # preprocessor3 = Preprocessor(df['Summary'].tolist())
        # df['Summary'] = preprocessor3.preprocess()
        #
        # preprocessor4 = Preprocessor(df['Reviews'].tolist())
        # df['Reviews'] = preprocessor4.preprocess()

        # Encode genre labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Genres'].apply(lambda x: ' '.join(x)))
        X = np.array(df['Summary'].values)

        return X, y
