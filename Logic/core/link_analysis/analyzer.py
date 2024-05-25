from .graph import LinkGraph
# from .indexer.indexes_enum import Indexes
# from .indexer.index_reader import Index_reader
import json
import random


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        self.hubs = [movie["id"] for movie in self.root_set]
        for movie in self.root_set:
            stars = movie["stars"]
            if stars:
                self.authorities.extend(stars)
        for movie in self.root_set:
            movie_id = movie["id"]
            stars = movie["stars"]
            self.graph.add_node(movie_id)
            if stars:
                for star in stars:
                    self.graph.add_node(star)
                    self.graph.add_edge(movie_id, star)
    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            #TODO
            if movie["stars"]:
                for star in movie["stars"]:
                    if star not in self.authorities:
                        if movie["id"] in self.hubs:
                            self.graph.add_node(star)
                            self.graph.add_edge(movie["id"], star)
                            self.authorities.append(star)
                            print(star)
            if movie["id"] not in self.hubs:
                if movie["stars"]:
                    for star in movie["stars"]:
                        if star in self.authorities and movie["id"] not in self.hubs:
                            self.graph.add_node(movie["id"])
                            self.graph.add_edge(movie["id"], star)
                            self.hubs.append(movie["id"])
                            print(movie["id"])

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = {actor: 1 for actor in self.authorities}
        h_s = {hub: 1 for hub in self.hubs}

        for _ in range(num_iteration):
            new_a_s = {}
            new_h_s = {}

            # Update authority scores
            for actor in a_s:
                new_a_score = sum(h_s[hub] for hub in self.graph.get_predecessors(actor))
                new_a_s[actor] = new_a_score

            # Update hub scores
            for hub in h_s:
                new_h_score = sum(a_s[actor] for actor in self.graph.get_successors(hub))
                new_h_s[hub] = new_h_score

            # Normalization
            a_norm = sum(new_a_s.values())
            h_norm = sum(new_h_s.values())
            a_s = {k: v / a_norm for k, v in new_a_s.items()}
            h_s = {k: v / h_norm for k, v in new_h_s.items()}

        # Sort and get top results
        if max_result is None:
            sorted_actors = sorted(a_s, key=a_s.get, reverse=True)
            sorted_movies = sorted(h_s, key=h_s.get, reverse=True)
        else:
            sorted_actors = sorted(a_s, key=a_s.get, reverse=True)[:max_result]
            sorted_movies = sorted(h_s, key=h_s.get, reverse=True)[:max_result]

        return sorted_actors, sorted_movies


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open("/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/Logic/Copy of IMDB_crawled.json", "r") as f:
        corpus = json.load(f)  # TODO: it shoud be your crawled data
    root_set_size = 100
    root_set = random.sample(corpus, root_set_size)  # TODO: it shoud be a subset of your corpus

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
