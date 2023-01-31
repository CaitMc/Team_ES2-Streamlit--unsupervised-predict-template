"""
Functions to build EDA page in streamlit app
"""
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from utils import data_loader as dl
import warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")
sns.set(font_scale=1)
sns.set_style("white")

# Load data
train_df = dl.load_dataframe('resources/data/ratings.csv', index=None)
movies_df = dl.load_dataframe('resources/data/movies.csv', index='movieId')
imdb_df = dl.load_dataframe('resources/data/imdb_data.csv', index=None)

# Functions
# Genres

# function that hasn't found a use yet
def feat_extractor(df, col):
    """
    returns a list of all unique features in a DataFrame columns separated by "|"
    """
    df.fillna("", inplace=True)
    feat_set = set()
    for i in range(len(df[f'{col}'])):
        for feat in df[f'{col}'].iloc[i].split('|'):
            feat_set.add(feat)
    return sorted([feat for feat in feat_set if feat != ""])

@st.cache(allow_output_mutation=True)
def feature_frequency(df, column):
    """
    Function to count the number of occurences of metadata such as genre
    Parameters
    ----------
        df (DataFrame): input DataFrame containing movie metadata
        column (str): target column to extract features from
    Returns
    -------

    """
    # Creat a dict to store values
    df = df.dropna(axis=0)
    genre_dict = {f'{column}': list(),
                 'count': list(),}
    # Retrieve a list of all possible genres
    print('retrieving features...')
    for movie in range(len(df)):
        gens = df[f'{column}'].iloc[movie].split('|')
        for gen in gens:
            if gen not in genre_dict[f'{column}']:
                genre_dict[f'{column}'].append(gen)
    # count the number of occurences of each genre
    print('counting...')
    for genre in genre_dict[f'{column}']:
        count = 0
        for movie in range(len(df)):
            gens = df[f'{column}'].iloc[movie].split('|')
            if genre in gens:
                count += 1
        genre_dict['count'].append(count)

        # Calculate metrics
    data = pd.DataFrame(genre_dict)
    return data

def feature_count(df, column):
    """
    Returns a barplot showing the number of movies per genre
    Parameters
    ----------
        df (DataFrame): input dataframe containing genre frequency
        column (str): target column to plot
    Returns
    -------
        barplot (NoneType): barplot visual
    Example
    -------
    """
    plt.figure(figsize=(10,6))
    ax = sns.barplot(y = df[f'{column}'], x = df['count'], palette='brg', orient='h')
    plt.title(f'Number of Movies Per {column[:-1]}', fontsize=14)
    plt.ylabel(f'{column}')
    plt.xlabel('Count')
    st.pyplot()

    #mean_ratings = pd.DataFrame(train_df.join(movies_df, on='movieId', how='left').join(imdb_df, on = 'movieId', how = 'left').groupby(['movieId'])['rating'].mean())

@st.cache(allow_output_mutation=True)
def mean_calc(feat_df, ratings = train_df, movies = movies_df, metadata = imdb_df, column = 'genres'):
    """
    Function that calculates the mean ratings of a feature
    Parameters
    ----------
        feat_df (DataFrame): input df containing feature data eg genres
        ratings (DataFrame): user-item interaction matrix
        movies (DataFrame): input df containing item names and genres column
        metadata (DataFrame): input df containing imdb metadata
        column (str): target column to calculate means
    Returns
    -------
        means (list): list of means for each genre
    Example
    -------
        >>>genres['mean_rating'] = mean_calc(genres)

    """
    mean_ratings = pd.DataFrame(ratings.join(movies, on='movieId', how='left').groupby(['movieId'])['rating'].mean())
    movie_eda = movies.copy()
    movie_eda = movie_eda.join(mean_ratings, on = 'movieId', how = 'left')

    # Exclude missing values
    movie_eda = movie_eda
    movie_eda2 = movie_eda[movie_eda['rating'].notnull()]

    means = []
    for feat in feat_df[f'{column}']:
        mean = round(movie_eda2[movie_eda2[f'{column}'].str.contains(feat)]['rating'].mean(),2)
        means.append(mean)
    return means
    genres['mean_rating'] = mean_calc(genres)
    genres.sort_values('mean_rating', ascending=False).head(5)


    # function that hasn't found a use yet
def feat_extractor(df, col):
    """
    returns a list of all unique features in a DataFrame columns separated by "|"
    """
    df.fillna("", inplace=True)
    feat_set = set()
    for i in range(len(df[f'{col}'])):
        for feat in df[f'{col}'].iloc[i].split('|'):
            feat_set.add(feat)
    return sorted([feat for feat in feat_set if feat != ""])

def genre_frequency(df):
    """
    docstring
    """
    # Creat a dict to store values
    genre_dict = {'genre': list(),
                 'count': list(),}
    # Retrieve a list of all possible genres
    for movie in range(len(df)):
        gens = df['genres'].iloc[movie].split('|')
        for gen in gens:
            if gen not in genre_dict['genre']:
                genre_dict['genre'].append(gen)
    # count the number of occurences of each genre
    for genre in genre_dict['genre']:
        count = 0
        for movie in range(len(df)):
            gens = df['genres'].iloc[movie].split('|')
            if genre in gens:
                count += 1
        genre_dict['count'].append(count)

        # Calculate metrics
    data = pd.DataFrame(genre_dict)
    return data
    data = genre_frequency(movies_df)

