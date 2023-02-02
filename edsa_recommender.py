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
import joblib,os
from PIL import Image
# Data handling dependencies
import pandas as pd
import numpy as np

# Data Visulization
import matplotlib.pyplot as plt

# Custom Libraries
from utils import data_loader as dl
from eda import eda_functions as eda
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
#from recommenders.content_based2 import recommend
st.set_option('deprecation.showPyplotGlobalUse', False)
import warnings
warnings.simplefilter(action='ignore')

# Data Loading


# Loading a css stylesheet
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("resources/css/style.css")

# App declaration
def main():
    
    page_options = ["Recommender System","Introduction", "Exploratory Data Analysis","Solution Overview","Contact Us"]

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    #page_options = ["Recommender System", "Introduction", "Exploratory Data Analysis", "Solution Overview"]

################################################################################
################################ MODEL #########################################
################################################################################

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    #page_selection = st.sidebar.selectbox("Select Page", page_options)
    page_selection = st.sidebar.selectbox("Select Page", page_options)
    if page_selection == "Recommender System":
        title_list = dl.load_movie_titles('resources/data/movies.csv')
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        #st.image('resources/imgs/image_header.png')
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        # movie_list = pd.merge(df_train, title_list, on = 'movieId', how ='left').groupby('title')['ratings'].mean()
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
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

################################################################################
################################ Solution Overview #############################
################################################################################

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.markdown(open('resources/markdown/solution.md').read(), unsafe_allow_html=True)
        st.image('resources/imgs/models2.png')

################################################################################
################################ EDA ###########################################
################################################################################

    # ------------- EDA -------------------------------------------
    if page_selection == "Exploratory Data Analysis":
        page_options_eda = ["User Interactions", "Movies", "Genres",]
        page_selection_eda = st.selectbox("Select Feature", page_options_eda)
        if page_selection_eda == "User Interactions":
            st.sidebar.markdown(open('resources/markdown/eda/userint.md').read(), unsafe_allow_html=True)

        # Most Active
            st.subheader("Most Active Users")
            top_user = st.checkbox('Include top user',value=False)

            ## include top user
            if top_user == True:
                image1 = Image.open("resources/imgs/rating_graph_user72315 (1).png")
                st.image(image1)
                
            else:
                image2 = Image.open('resources/imgs/rating_graph_no_user72315.png')
                st.image(image2)

            st.write("User 72315 has rated an extreme number of movies relative to other users. For EDA purposes, this user can be removed above to make interpretation easier.")

        # Ratings Distribution
            st.subheader('Ratings Distribution')
            image3 = Image.open('resources/imgs/Distribution_of_Ratings.png')
            st.image(image3)
            st.write(open('resources/markdown/eda/ratings_dist.md').read(), unsafe_allow_html=True)

        # Rating v number of ratings
            st.subheader('Ratings trends')
            image4 = Image.open('resources/imgs/Mean_Ratings_by_Number_of_Ratings.png')
            st.image(image4)
            st.write('it seems like The more ratings a movie has, the more highly it is likely to be rated. This confirms our intuitive understanding that the more highly rated a movie is, the more likely is that viewers will recommend the movie to each other. In other words, people generally try to avoid maing bad recommendations')

        if page_selection_eda == "Movies":
            st.sidebar.markdown(open('resources/markdown/eda/movies.md').read(), unsafe_allow_html=True)
            st.subheader('Best and Worst Movies by Genre')
            image5 = Image.open('resources/imgs/top_15 best_movie_Ratings.png')
            st.image(image5)            
            st.write('By filtering movies with less than 10000 ratings, we find that the most popular movies are unsurprising titles. The Shawshank Redemption and The Godfather unsurprisingly top the list. What is interesting is that Movies made post 2000 do not feature often. Do users have a preference to Older movies?')
            image6 = Image.open('resources/imgs/top_15_worst_movie_Ratings.png')
            st.image(image6)
            st.write('Obviously, users did not like Battlefield too much and with 1200 ratings, they really wanted it to be known. It is interesting how many sequels appear in the list')


        if page_selection_eda == "Genres":
            st.sidebar.markdown(open('resources/markdown/eda/genres.md').read(), unsafe_allow_html=True)
            st.subheader('Genre Distribution')
            image7 = Image.open('resources/imgs/Number_of_movies_per_genres.png')
            st.image(image7)
            st.write('Drama is the most frequently occuring genre in the database. Approximately 5000 movies have missing genres. We can use the IMDB and TMDB IDs together with the APIs to fill missing data. Further, IMAX is not a genre but rather a proprietary system for mass-viewings.')
            st.write('The above figure does not tell us anything about the popularity of the genres, lets calculate a mean rating and append it to the Data')
            movies_df = dl.load_dataframe('resources/data/movies.csv', index=None)
            genres= eda.feature_frequency(movies_df, 'genres')
            genres['mean_rating']=eda.mean_calc(genres)
            show_data = st.checkbox('Show raw genre data?')
            if show_data:
                st.write(genres.sort_values('mean_rating', ascending=False))
            st.write('Film-Noir describes Hollywood crime dramas, particularly those that emphasize cynical attitudes and sexual motivations. The 1940s and 1950s are generally regarded as the "classic period" of American film-noir. These movies have the highest ratings but this may be as a result of its niche audence. The same logic can be applied to IMAX movies, as such, we will only include genres with a count of 500 or more.')
            image8 = Image.open('resources/imgs/Mean_Rating_Per_Genre.png')
            st.image(image8)
            st.write('The scores are almost evenly distributed with the exceptions of Documentaries, War, Drama, Musicals, and Romance and Thriller, Action, Sci-Fi, and Horror, which rate higher than average and below average respectively.')

################################################################################
################################ Introduction ##################################
################################################################################

    if page_selection == "Introduction":
        st.sidebar.markdown(open('resources/markdown/introduction/contrib.md').read(), unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Team ES2 Data Flex</h1>", unsafe_allow_html=True)
        st.image('resources/imgs/team_logo.jpg',use_column_width=True)
       
        info_pages = ["Select Option", "General Information"]
        info_page_selection = st.selectbox("", info_pages)

        if info_page_selection == "Select Option":
            st.info("Welcome! Select an option from the menu above to get started.")

        if info_page_selection == "General Information":
            st.info("Read more about the project and the data that was used to solve the problem at hand.")
            st.markdown(open('resources/markdown/introduction/general_information/intro.md').read(), unsafe_allow_html=True)

            definitions = st.checkbox("Show definitions")
            see_raw = st.checkbox("Show data")

            if definitions:
                st.write(open('resources/markdown/introduction/general_information/data_def.md', encoding='utf8').read(), unsafe_allow_html=True)
            if see_raw:
                st.write(dl.load_dataframe('resources/data/ratings.csv', index='userId').head(10))
                st.write(dl.load_dataframe('resources/data/movies.csv',index='movieId').head(10))
#################################################################################
############################## contact us page ##################################
#################################################################################
        
     # Building out the "Contact Us" page
    if page_selection == "Contact Us":
        st.image('resources/imgs/team_logo.jpg',use_column_width=True)
        if page_selection == "Contact Us":
            with st.form("form1", clear_on_submit=True):
                name = st.text_input("Enter full name")
                email = st.text_input("Enter email")
                message = st.text_area("Message")

                submit = st.form_submit_button("Submit Form")
                
                
################################################################################
################################ ###########################################
################################################################################
#if page_selection == "hybrid recommender system":
#        movies_dict = pickle.load(open('resources/pickel_files/movie_dict.pkl', 'rb'))
#       movies2 = pd.DataFrame(movies_dict)
#
#        similarity = pickle.load(open('resources/pickel_files/similarity.pkl', 'rb'))
#        st.title('Movie Recommender System')
#
#        selected_movie_name = st.selectbox(
#            'How would you like to be contacted?',
#            movies2['title'].values
#        )
#
#        if st.button('Recommend'):
#            names, posters = recommend(selected_movie_name)
#            col1, col2, col3, col4, col5 = st.columns(5)
#            with col1:
#                st.text(names[0])
#                st.image(posters[0])
#            with col2:
#                st.text(names[1])
#                st.image(posters[1])
#            with col3:
#                st.text(names[2])
#                st.image(posters[2])
#            with col4:
#                st.text(names[3])
#                st.image(posters[3])
#            with col5:
#                st.text(names[4])
#                st.image(posters[4])

if __name__ == '__main__':
    main()
