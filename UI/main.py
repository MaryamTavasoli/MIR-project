import streamlit as st
import sys
import time
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

    # Feedback section
    st.sidebar.header("Feedback")
    if "feedback" not in st.session_state:
        st.session_state.feedback = ""

    feedback = st.sidebar.text_area("Your Feedback:", value=st.session_state.feedback, key="feedback_area")

    if st.sidebar.button("Submit Feedback"):
        save_feedback(feedback)
        st.session_state.feedback = ""
        st.sidebar.success("Thank you for your feedback!")


    # Navigation to introduction page
    st.sidebar.markdown("---")

    if st.sidebar.button("About"):
        show_introduction()


def save_feedback(feedback):
    feedback_file = "/Users/maryamtavasoli/Desktop/IMDb-IR-System-main-phase2/UI/user_feedback.txt"  # Name of the feedback file
    with open(feedback_file, "a", encoding="utf-8") as file:
        file.write(feedback + "\n")


def show_introduction():
    st.title("About Movie Search Engine")
    st.write("""
    Welcome to the Movie Search Engine, a simple search application for IMDb movies. 
    This application allows you to search through the IMDb dataset and find the most relevant movie based on your search terms.
    """)

    st.header("How It Works")
    st.write("""
    The search engine uses advanced indexing and ranking algorithms to provide accurate search results. 
    You can adjust search parameters such as weights for different features and search methods to refine your results.
    """)

    st.header("About Us")
    st.write("""
    This project is a collaborative effort between the MIR Team at Sharif University and Maryam Tavasoli. 
    Our goal is to provide a user-friendly and efficient movie search experience using modern information retrieval techniques.
    """)


def main():
    st.title("Movie Search Engine")
    st.write(
        "This is a simple search engine for IMDb movies. You can search through the IMDb dataset and find the most relevant movie to your search terms."
    )
    st.markdown(
        '<span style="color:red">Developed By: MIR Team at Sharif University</span>',
        unsafe_allow_html=True,
    )

    # Sidebar for advanced search options
    with st.sidebar:
        st.header("Advanced Search")
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )

        weight_stars = st.slider("Weight for Stars", 0.0, 1.0, 0.3, 0.1, key="stars_weight")
        weight_genres = st.slider("Weight for Genres", 0.0, 1.0, 0.3, 0.1, key="genres_weight")
        weight_summary = st.slider("Weight for Summary", 0.0, 1.0, 0.4, 0.1, key="summary_weight")

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram"), key="search_method"
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
                key="smoothing_method"
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="alpha_slider"
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="alpha_slider"
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key="lambda_slider"
                )

        # Color picker for highlighted text
        highlight_color = st.color_picker("Pick a highlight color", value="#00FF51", key="color_picker")

        # Filter results interface
        num_filter_results = st.number_input(
            "Number of filtered results to show", min_value=1, max_value=20, value=10, step=1, key="filter_results"
        )
        filter_button_key = "filter_button"  # Unique key for filter button
        filter_button = st.button("Filter Results", key=filter_button_key)

        if filter_button:
            if num_filter_results > search_max_num:
                st.warning(
                    f"Filtered results cannot exceed maximum number of search results ({search_max_num}). Setting filtered results to 1."
                )
                num_filter_results = 1

    # Main search interface
    search_term = st.text_input("Enter your search term:")
    search_button_key = "search_button"  # Unique key for search button
    search_button = st.button("Search", key=search_button_key)

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
        highlight_color,
    )


if __name__ == "__main__":
    main()
