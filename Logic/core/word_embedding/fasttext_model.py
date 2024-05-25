import string
import fasttext
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
import numpy as np


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=None, lower_case=True,
                    punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if stopwords_domain is None:
        stopwords_domain = ["this", "that", "about", "whom", "being",
                            "where", "why", "had", "should", "each"]
    if lower_case:
        text = text.lower()
    words = word_tokenize(text)
    if punctuation_removal:
        words = [word for word in words if word not in string.punctuation]
    if stopword_removal:
        stop_words = set(stopwords.words('english')).union(set(stopwords_domain))
        words = [w for w in words if w not in stop_words]
    words = [w for w in words if len(w) >= minimum_length]
    preprocessed_text = ' '.join(words)
    return preprocessed_text


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, preprocessor, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None

    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        print("training the FastText model...")
        with open('fasttext_train.txt', 'w') as f:
            for textlist in texts:
                for text in textlist:
                    f.write(text + ' ')
                f.write('\n')
        self.model = fasttext.train_unsupervised('fasttext_train.txt', model=self.method)

    def get_query_embedding(self, query, do_preprocess):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = preprocess_text(query)

        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        # TODO
        vec1 = self.model.get_word_vector(word1)
        vec2 = self.model.get_word_vector(word2)
        vec3 = self.model.get_word_vector(word3)

        # Perform vector arithmetic
        # TODO
        analogy_vector = vec2 - vec1 + vec3
        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # TODO
        vocabulary = self.model.words
        word_vectors = {w: self.model.get_word_vector(w) for w in vocabulary}
        # Exclude the input words from the possible results
        # TODO
        excluded_words = {word1, word2, word3}
        # Find the word whose vector is closest to the result vector
        # TODO
        len_analogy = np.linalg.norm(analogy_vector)
        closest_word = None
        max_dot_product = -float("inf")
        for word, vector in word_vectors.items():
            if word not in excluded_words:
                dot_product = np.dot(vector, analogy_vector) / len_analogy
                if dot_product > max_dot_product:
                    max_dot_product = dot_product
                    closest_word = word

        return closest_word

    def save_model(self, path):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        elif mode == 'load':
            self.load_model(path)
        elif mode == 'save' and save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')

    path = '/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/core/indexes/preprocessed_index4.json'
    ft_data_loader = FastTextDataLoader(path)

    X, y = ft_data_loader.create_train_data()

    ft_model.train(X)
    ft_model.prepare(None, mode="save", save=True)
    # ft_model.prepare(None, mode="load")
    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "man"
    word2 = "king"
    word3 = "woman"
    print(
        f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
