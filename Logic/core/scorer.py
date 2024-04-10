import math


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            df = len(self.index.get(term, {}))
            idf = math.log(self.N / (df + 0.5))
            self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        # TODO
        query_tfs = {}
        for term in query:
            query_tfs[term] = query_tfs.get(term, 0) + 1
        return query_tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        scores = {}
        query_tfs = self.get_query_tfs(query)
        for document_id in self.get_list_of_documents(query):
            score = self.get_vector_space_model_score(query, query_tfs, document_id, method[4:7], method[:3])
            scores[document_id] = score
        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        # TODO
        score = 0
        norm_d = 0
        norm_q = 0
        for term, tf in query_tfs.items():
            idf = self.get_idf(term)
            idf_q = 1
            idf_d = 1
            tf_q = 1
            tf_d = self.index.get(term, {}).get(document_id, 0)
            if query_method[0] == 'l':
                tf_q = 1 + math.log(tf)
            elif query_method[0] == 'n':
                tf_q = 1
            if document_method[0] == 'l':
                if tf_d >= 1:
                    tf_d = 1 + math.log(tf_d)
                else:
                    tf_d = 0
            elif document_method[0] == 'n':
                tf_d = 1
            if query_method[1] == 't':
                idf_q = idf
            if document_method[1] == 't':
                idf_d = idf
            score += tf_q * idf_q * tf_d * idf_d
            norm_q += (idf_q * tf_q) * (idf_q * tf_q)
        if document_method[2] == 'c':
            for key, value in self.index.items():
                for key1, value1 in value.items():
                    if key1 == document_id:
                        idf = 1
                        tf = 1
                        if document_method[0] == 'l':
                            tf = 1 + math.log(self.index.get(key, {}).get(document_id, 0))
                        if document_method[1] == 't':
                            idf = self.get_idf(key)
                        norm_d += (tf * idf) * (tf * idf)
        if document_method[2] == 'n':
            norm_d = 1
        elif document_method[2] == 'c':
            norm_d = 1 / (math.sqrt(norm_d))
        if query_method[2] == 'n':
            norm_q = 1
        elif query_method[2] == 'c':
            norm_q = 1 / (math.sqrt(norm_q))
        return score * norm_d * norm_q

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        scores = {}
        k1 = 1.2
        b = 0.75
        for document_id in self.get_list_of_documents(query):
            score = self.get_okapi_bm25_score(query, document_id, average_document_field_length, document_lengths, k1,
                                              b)
            scores[document_id] = score
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths, k1, b):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        k1 : float
            Parameter k1 for tuning BM25.
        b : float
            Parameter b for tuning BM25.
        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO
        score = 0.0
        dl = document_lengths[document_id]
        query_tfs = self.get_query_tfs(query)
        for term, tf in query_tfs.items():
            idf = self.get_idf(term)
            tf_d = self.index.get(term, {}).get(document_id, 0)
            numerator = tf_d * (k1 + 1)
            denominator = tf_d + (k1 * ((1 - b) + (b * (dl / average_document_field_length))))
            score += idf * (numerator / denominator)
        return score
