import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        with open("stopwords.txt", 'r', encoding='utf-8') as file:
            stopwords = file.read().splitlines()
        stop_words = set(stopwords)
        query_words = word_tokenize(query)
        filtered_query_words = [word for word in query_words if word.lower() not in stop_words]

        return ' '.join(filtered_query_words)

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        # Remove stop words from the query
        query = self.remove_stop_words_from_query(query)
        # Split the document and query into words
        doc_words = word_tokenize(doc)
        query_words = word_tokenize(query)
        l = len(query_words)
        snnipet_exist_another_token = []
        snnipets = []
        for token in query_words:
            found = False
            for i in range(len(doc_words)):
                if doc_words[i] == token:
                    found = True
                    doc_words[i] = "***" + doc_words[i] + "***"
                    x=i-3
                    y=i+3
                    if i-3<0:
                      x=0
                    elif i+3>len(doc_words)-1:
                      y=len(doc_words)-1
                    snnipets.append(' '.join(doc_words[x:y]))
                    for token1 in query_words:
                        if token1 != token and token1 in doc_words[x:y]:
                            snnipet_exist_another_token.append(' '.join(doc_words[x:y]))
            if not found:
                not_exist_words.append(token)
        if len(snnipet_exist_another_token) == 0:
            for i in range(len(snnipets)):
                print(snnipets[i])
                final_snippet += snnipets[i]
                if i < len(snnipets) - 1:
                    final_snippet += "..."
        else:
            for i in range(len(snnipet_exist_another_token)):
                final_snippet += snnipet_exist_another_token[i]
                if i < len(snnipet_exist_another_token) - 1:
                    final_snippet += "..."

        return final_snippet, not_exist_words
