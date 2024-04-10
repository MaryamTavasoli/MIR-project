class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        # TODO: Create shingle here
        for i in range(len(word) - k + 1):
            shingle = word[i:i + k]
            shingles.add(shingle)
        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))
        return intersection / union if union != 0 else 0

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.

        for document in all_documents:
            words = document.split()
            for word in words:
                shingles = self.shingle_word(word)
                all_shingled_words[word] = shingles
                word_counter[word] = word_counter.get(word, 0) + 1

        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        scores = []
        for candidate, shingles in self.all_shingled_words.items():
            jaccard_score = self.jaccard_score(self.all_shingled_words[word], shingles)
            score_tuple = (jaccard_score, candidate)
            scores.append(score_tuple)

        sorted_candidates = sorted(scores, key=lambda x: x[0], reverse=True)

        for score, candidate in sorted_candidates[:5]:
            top5_candidates.append(candidate)

        return top5_candidates

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        # TODO: Do spell correction here.
        query_words =query.split()

        corrected_query = []
        for word in query_words:
            if word in self.word_counter:
                corrected_query.append(word)
            else:
                nearest_words = self.find_nearest_words(word)
                if nearest_words:
                    corrected_query.append(nearest_words[0])
                else:
                    corrected_query.append(word)

        final_result = ' '.join(corrected_query)
        return final_result
