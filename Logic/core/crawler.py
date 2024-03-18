from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import requests
import pandas as pd

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
       'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    }

    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = set()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        crawled_data = {
            "crawled": list(self.crawled),
        }
        not_crawled_data = {
            "not_crawled": list(self.not_crawled),
        }

        with open('IMDB_crawled.json', 'w') as f:
            json.dump(crawled_data, f)
        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(not_crawled_data, f)
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
             self.crawled = set(json.load(f)["crawled"])

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = deque(json.load(f)["not_crawled"])

        self.added_ids = set(json.load(f)["added_ids"])

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # Make a GET request to the URL
        response = requests.get(URL,headers=self.headers)

        # Return the response
        return response

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        # response = self.crawl(self.top_250_URL)
        response=self.crawl(self.top_250_URL)
        # print(response,"1")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            movie_links = soup.find_all('a',{'class_','ipc-title-link-wrapper'})
            for link in movie_links:
              if link['href'].startswith('/title'):
                movie_url= "https://www.imdb.com"+link['href']
                movie_id = self.get_id_from_URL(movie_url)
                if movie_url not in self.crawled and movie_id not in self.added_ids:
                    self.not_crawled.append(movie_url)
                    self.added_ids.add(movie_id)

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO:
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()

        futures = []
        crawled_counter = 0

        # with ThreadPoolExecutor(max_workers=20) as executor:
        while self.not_crawled and crawled_counter < self.crawling_threshold:
          URL = self.not_crawled.popleft()
          crawled_counter+=1
          # print(URL,"  ",crawled_counter)
          futures.append(self.crawl_page_info(URL))
          if not self.not_crawled:
            if futures is not None:
              wait(futures)
              futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.

        Parameters
        ----------
        URL: str
            The URL of the site
        """
        # print("new iteration")
        response=self.crawl(URL)
        # print(response,2)
        if response.status_code == 200:
          movie=self.get_imdb_instance()
          movie=self.extract_movie_info(response,movie,URL)
          # print(movie)
          for key, value in movie.items():
             self.crawled.add((key, value))
          # print(self.crawled)
          if movie['related_links'] is not None:
            for link in movie['related_links']:
              if link not in self.added_ids:
                  self.not_crawled.append(link)
                  self.added_ids.add(self.get_id_from_URL(link))

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        soup = BeautifulSoup(res.content, 'html.parser')
        # print(soup)
        movie['title'] = self.get_title()
        # print(movie['title'])
        movie['first_page_summary'] = self.get_first_page_summary()
        movie['release_year'] = self.get_release_year()
        movie['mpaa'] = self.get_mpaa()
        movie['budget'] = self.get_budget()
        movie['gross_worldwide'] = self.get_gross_worldwide()
        movie['directors'] = self.get_director()
        movie['writers'] = self.get_writers()
        movie['stars'] = self.get_stars()
        movie['related_links'] = self.get_related_links()
        movie['genres'] = self.get_genres()
        movie['languages'] = self.get_languages()
        movie['countries_of_origin'] = self.get_countries_of_origin()
        movie['rating'] = self.get_rating()
        # response=requests.get(self.get_summary_link(URL),headers=self.headers)
        # soup1= BeautifulSoup(response.content, 'html.parser')
        movie['summaries'] = self.get_summary()
        movie['synopsis'] = self.get_synopsis()
        # response1=requests.get(self.get_review_link(URL),headers=self.headers)
        # soup2= BeautifulSoup(response1.content, 'html.parser')
        movie['reviews'] = self.get_reviews_with_scores()
        return movie
    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = url.split('/')[4]
            summary_link = f'https://www.imdb.com/title/{movie_id}/plotsummary'
            return summary_link
        except:
            print("failed to get summary link")

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = url.split('/')[4]
            review_link = f'https://www.imdb.com/title/{movie_id}/reviews'
            return review_link
        except:
            print("failed to get review link")

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title_element = soup.find('div', class_='sc-d8941411-1')
            print(title_element)
            if title_element:
              return  title_element.text.strip().split(':')[-1].strip()
            else:
              return"failed to get title"

        except:
            print("failed to get title")

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            summary_element = soup.find('span', class_='sc-466bb6c-2')
            if summary_element:
              summary = summary_element.text.strip()
              return summary
            return None
        except:
            print("failed to get first page summary")

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            director_label = soup.find('span', attrs={'aria-label': 'Director'})
            if director_label:
                director_container = director_label.find_parent('div', class_='ipc-metadata-list-item')
                if director_container:
                    directors = [director.text.strip() for director in director_container.find_all('a')]
                    return directors
        except:
            print("failed to get director")
            return None

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            star_elements = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item--link')
            if star_elements:
                stars = [star.text.strip() for star in star_elements]
                return stars
        except:
            print("failed to get stars")

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
           writer_ul = soup.find('ul', class_='ipc-inline-list')
           if writer_ul:
              writer_elements = writer_ul.find_all('a', class_='ipc-metadata-list-item__list-content-item')
              writers = [writer.text.strip() for writer in writer_elements]
              return writers
           return None
        except:
            print("failed to get writers")

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related_links = []
            # Find the container div for related links
            shoveler_div = soup.find('div', class_='ipc-shoveler')
            if shoveler_div:
               anchor_tags = shoveler_div.find_all('a', class_='ipc-metadata-list-item__list-content-item--link')
               related_links = [a.get('href') for a in anchor_tags]
            return related_links
        except:
            print("failed to get related links")
            return None

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            summary_list = []
            # Find all <li> elements containing the summary
            summary_items = soup.find_all('li', class_='ipc-metadata-list__item')
            for item in summary_items:
                summary_text = item.find('div', class_='ipc-html-content').text.strip()
                summary_list.append(summary_text)
            return summary_list
        except:
            print("failed to get summary")

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            synopsis_list = []
            synopsis_div = soup.find('div', class_='ipc-html-content-inner-div')
            synopsis_text = synopsis_div.text.strip()
            synopsis_list.append(synopsis_text)
            return synopsis_list
        except:
            print("failed to get synopsis")
            return None

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews_with_scores = []
            review_divs = soup.find_all('div', class_='lister-item-content')
            for div in review_divs:
               # Find the review text
               review_text = div.find('div', class_='text').text.strip()
               # Find the review score
               score_element = div.find('span', class_='rating-other-user-rating')
               score = score_element.find('span').text.strip() if score_element else None
               # Append review and score to the list
               reviews_with_scores.append([review_text, score])
            return reviews_with_scores
        except:
            print("failed to get reviews")
            return None

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            genres = []
            genre_labels = soup.find_all('span', class_='ipc-metadata-list-item__label', text='Genre')
            for label in genre_labels:
              parent_element = label.find_parent('div', class_='ipc-metadata-list-item__content-container')
              genre_elements = parent_element.find_all('a', class_='ipc-metadata-list-item__list-content-item')
              genres.append([genre.text.strip() for genre in genre_elements])
            return genres
        except:
            print("Failed to get generes")

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # Find the span element containing the rating
            rating_element = soup.find('span', class_='sc-bde20123-1 cMEQkK')
            # Extract the text content of the rating element
            rating = rating_element.text.strip() if rating_element else None
            return rating
        except:
            print("failed to get rating")

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            mpaa_element = soup.find('span', class_='ipc-metadata-list-item__list-content-item')
            mpaa = mpaa_element.text.strip() if mpaa_element else None
            return mpaa
        except:
            print("failed to get mpaa")
            return None

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            release_element = soup.find('a', class_='ipc-metadata-list-item__list-content-item')
            release_info = release_element.text.strip() if release_element else None
            release_year = release_info.split()[-1]
            return release_year
        except:
            print("failed to get release year")
            return None

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            container = soup.find('div', class_='ipc-metadata-list-item__content-container')
            language_elements = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
            languages = [element.text.strip() for element in language_elements]
            return languages
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
             container = soup.find('div', class_='ipc-metadata-list-item__content-container')
             country_elements = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
             countries = [element.text.strip() for element in country_elements]
             return countries
        except:
            print("failed to get countries of origin")

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            section_part= soup.find('section',{'data-testid': 'Boxoffice'})
            div_part=section_part.find('div',{'data-testid':'title-boxoffice-section'})
            li_part=div_part.find('li',{'data-testid':'title-boxoffice-budget'})
            span_part=li_part.find('span',{'class':'ipc-metadata-list-item__list-content-item'})
            return span_part.text.strip()
        except:
            print("failed to get budget")

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            container = soup.find('div', {'data-testid': 'Boxoffice'})
            gross_element = container.find('li', {'data-testid': 'title-boxoffice-cumulative-gross'})
            gross_worldwide = gross_element.find('span', {'class': 'ipc-metadata-list-item__list-content-item'}).text.strip()
            return gross_worldwide
        except:
            print("failed to get gross worldwide")

def main():
    imdb_crawler = IMDbCrawler()
    # imdb_crawler.write_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()

    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
    # }
    # url="https://www.imdb.com/title/tt0111161/"
    # Response=requests.get(url,headers=headers)
    # soup=BeautifulSoup(Response.content,'html.parser')
    # # print(soup)
    # title_element = soup.find('div', class_='sc-d8941411-1')
    # if title_element:
    #     print(title_element.text.strip().split(':')[-1].strip())
    # else:
    #     print("failed to get title")



if __name__ == '__main__':
    main()
