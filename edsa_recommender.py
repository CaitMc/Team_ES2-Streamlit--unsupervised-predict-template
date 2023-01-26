"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
#import numpy as np
#import itertools
#import re
#import matplotlib.pyplot as plt
#import seaborn as sns

# Custom Libraries
from utils.data_loader import load_movie_titles
#from recommenders.collaborative_based import collab_model
#from recommenders.content_based import content_model
#from recommenders.content_based import content_model
#from recommenders.collaborative_based import yg_collab_model
#from recommenders.content_based import content_model_adj

## Data Loading
#title_list = load_movie_titles('resources/data/movies.csv')
#movies_and_ratings = pd.read_csv('resources/data/movies_and_ratings.csv')

# ===================================================================
# ======================== Preprocessing ============================
# ===================================================================
# ====The following datasets are required for this app to run,=======
# ===================Please load them all============================
# ===================================================================
#df_movies = pd.read_csv("resources/data/movies.csv")
#df_links = pd.read_csv('resources/data/links.csv')
#df_imdb = pd.read_csv('resources/data/imdb_data.csv')
#df_movies = pd.read_csv('resources/data/movies.csv')
#df_ratings = pd.read_csv('resources/data/ratings.csv')
#the df_ratings only contains data <= 2016, popular page is based on >2018 data. latest_ratings has added post 2018 data which is 11k long
#df_latest_ratings = pd.read_csv('resources/data/latest_ratings.csv')
# Do not use the training data. Too big. Ratings is the same, just smaller. Use that instead.
#df_train = pd.read_csv('resources/data/train.csv')
#df_tags = pd.read_csv('resources/data/tags.csv')


#________________________________________________________________________________________
# Find unique genres

# genres = pd.DataFrame(df_movies.genres.apply(lambda x: x.split('|')), columns=['genres'])
#
# unique_genres = sorted(list(set(itertools.chain.from_iterable(genres.genres))))
# # Remove uncommon genres
# unique_genres.remove('(no genres listed)')
# unique_genres.remove('IMAX')
# unique_genres.remove('Film-Noir')

#_______________________________________________________________________________________
## Release years

# Add a 'year' column to the movies dataframe that contains the release year of each movie (if available)
#years = df_movies.title.apply(lambda x: re.findall(r'\((.[\d]+)\)', x))
#df_movies['year'] = years.str[-1]
#df_movies.year = pd.to_numeric(df_movies.year).fillna(1800).astype(int)


#df_movies['genre_list'] = df_movies['genres'].str.split('|')
## Create dataframe containing only the title and genres
#movies_genres11 = df_movies[['title','genre_list']]

# Create expanded dataframe where each movie-genre combination is in a seperate row
#movies_genres11 = pd.DataFrame([(tup.title, d) for tup in movies_genres11.itertuples() for d in tup.genre_list],
#                             columns=['title', 'genres'])
#df_movies.year = pd.to_numeric(df_movies.year).fillna(1800).astype(int)

#_______________________________________________________________________________________
# Movie-genre combinations

# Create dataframe containing only the movieId and genres
#movies_genres = pd.DataFrame(df_movies[['movieId','genres']], columns=['movieId','genres'])

## Split genres seperated by "|" and create a list containing the genres allocated to each movie
#movies_genres.genres = movies_genres.genres.apply(lambda x: x.split('|'))

## Create expanded dataframe where each movie-genre combination is in a seperate row
#movies_genres = pd.DataFrame([(tup.movieId,d) for tup in movies_genres.itertuples() for d in tup.genres],
#                             columns=['movieId','genres'])
#_______________________________________________________________________________________
#combine tmdbid with web link to create movie link
#df_links['tmdbId_full_link'] = df_links['tmdbId'].apply(lambda x: "https://www.themoviedb.org/movie/"+str(x))

#df_movies = df_movies[df_movies['genres'] != '(no genres listed)']
#df_movies['genres'] = df_movies['genres'].str.split('|')
#df_movies['year'] = df_movies['title'].apply(lambda x: ' '.join(re.findall(r'\ \((\d+)\)$', str(x))))
#df_movies['year'] = pd.to_numeric(df_movies['year'])

#movie_ratings = pd.merge(df_latest_ratings[['userId','movieId','rating']], df_movies, on = 'movieId')

#movie_pref = movie_ratings[movie_ratings['year'] > 2018].groupby('title')\
#            .agg(num_ratings=('movieId', 'size'), average_rating=('rating', 'mean')).reset_index()

## Create dataframe containing only the title and genres
#m_genres = df_movies[['title','genres']]

# #Create expanded dataframe where each movie-genre combination is in a seperate row
#m_genres = pd.DataFrame([(tup.title,d) for tup in m_genres.itertuples() for d in tup.genres],
 #                            columns = ['title', 'genres'])
#m_genres = pd.merge(m_genres,movie_pref, on = 'title')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview", 'EDSA']

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    if page_selection == "EDSA":
        st.title("EDSA Overview")
        st.write("Describe your winning approach on this page")



    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
