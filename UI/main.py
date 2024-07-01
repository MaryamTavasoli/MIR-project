# Updated snippet incorporating suggestions

import streamlit as st
import sys
import time
import random
from enum import Enum

sys.path.append("../")
from Logic import utils
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes

# Initialize Snippet object
snippet_obj = Snippet()


# Define color enumeration for styling
class Color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    YELLOW = "#FFFF00"
    WHITE = "#FFFFFF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


# Get top movies by rank function
def get_top_x_movies_by_rank(x: int, results: list):
    path = "../Logic/core/indexes/"
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []

    for movie_id, movie_detail in document_index.index['documents'].items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index['documents'][movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


# Get summary with snippet function
def get_summary_with_snippet(movie_info, query, color):
    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={color}>{current_word_without_star}</font></b>",
                )
    return summary


# Search time display function
def search_time(start, end):
    st.success("Search took: {:.6f} milliseconds".format((end - start) * 1e3))


# Main search handling function
def search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        num_filter_results,
        color,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.markdown(f"**Top {num_filter_results} Actors:**")
            actors_ = ", ".join(top_actors)
            st.markdown(
                f"<span style='color:{color}'>{actors_}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

        st.markdown(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            with st.container():
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.markdown(
                        f"<b><font size='4'>Summary:</font></b> {get_summary_with_snippet(info, search_term, color)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")
                    if info["directors"]:
                        num_authors = len(info["directors"])
                    for j in range(num_authors):
                        st.text(info["directors"][j])

                with st.container():
                    st.markdown("**Stars:**")
                    num_authors = len(info["stars"])
                    stars = "".join(star + ", " for star in info["stars"])
                    st.text(stars[:-2])

                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        st.write("Genres:")
                        num_topics = len(info["genres"])
                        for j in range(num_topics):
                            st.markdown(
                                f"<span style='color:{color}'>{info['genres'][j]}</span>",
                                unsafe_allow_html=True,
                            )
                with card[1].container():
                    st.image(info["Image_URL"], use_column_width=True)

                st.divider()
        return

    if search_button:
        corrected_query = utils.correct_text(search_term, utils.all_documents)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        for i in range(len(result)):
            with st.container():
                card = st.columns([3, 1])
                info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
                with card[0].container():
                    st.title(info["title"])
                    st.markdown(f"[Link to movie]({info['URL']})")
                    st.write(f"Relevance Score: {result[i][1]}")
                    st.markdown(
                        f"<b><font size='4'>Summary:</font></b> {get_summary_with_snippet(info, search_term, color)}",
                        unsafe_allow_html=True,
                    )

                with st.container():
                    st.markdown("**Directors:**")
                    if info["directors"] is not None:
                        num_authors = len(info["directors"])
                        for j in range(num_authors):
                            st.text(info["directors"][j])

                with st.container():
                    st.markdown("**Stars:**")
                    if info["stars"] is not None:
                        num_authors = len(info["stars"])
                        stars = "".join(star + ", " for star in info["stars"])
                        st.text(stars[:-2])

                    topic_card = st.columns(1)
                    with topic_card[0].container():
                        st.write("Genres:")
                        if info["genres"] is not None:
                            num_topics = len(info["genres"])
                            for j in range(num_topics):
                                st.markdown(
                                    f"<span style='color:{color}'>{info['genres'][j]}</span>",
                                    unsafe_allow_html=True,
                                )
                with card[1].container():
                    st.image(info["Image_URL"], use_column_width=True)

                st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                    "search_results" in st.session_state
                    and len(st.session_state["search_results"]) > 0
            )


# Main function to run the app
def main():
    st.title("Movie Search Engine")
    st.write(
        "This is a simple search engine for IMDb movies. You can search through the IMDb dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:yellow">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    # Sidebar for advanced search options
    with st.sidebar:
        st.header("Advanced Search")
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=1
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox("Unigram Smoothing", ("laplace", "jm"))
            if unigram_smoothing == "jm":
                alpha = st.number_input(
                    "Enter alpha value (JM Smoothing)", min_value=0.0, max_value=1.0, value=0.5
                )
            lamda = st.number_input(
                "Enter lambda value (Dirichlet)", min_value=0.0, max_value=1.0, value=0.5
            )

        color = st.color_picker("Pick a color for text highlighting", "#00FF00")

        filter_button = st.button("Filter Results")

    search_term = st.text_input("Enter your search term:")
    search_button = st.button("Search")

    num_filter_results = slider_
    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        num_filter_results,
        color,
    )


if __name__ == "__main__":
    main()
