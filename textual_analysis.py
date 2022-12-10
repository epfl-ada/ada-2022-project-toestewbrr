''' 
Applied Data Analysis @ EPFL
Team: ToeStewBrr - Alexander Sternfeld, Marguerite Thery, Antoine Bonnet, Hugo Bordereaux 
Project: Love stories in movies
Dataset: CMU Movie Summary Corpus
'''

import os
import spacy
import nltk
import pandas as pd
import numpy as np

from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


#Load the spacy model
nltk.download('stopwords')
nltk.download('punkt')
nlp_spacy = spacy.load("en_core_web_lg")

# --------------- Embedding ----------------- #

def construct_descriptions_embeddings(df, nlp_spacy):
    ''' Compute the embeddings of all words in the character descriptions. '''
    # Keep a vocabulary dictionary of {word : embedding} pairs to avoid recomputing embeddings
    vocab = {}
    
    # Initialize the column with NaNs
    df['descriptions_embeddings'] = np.nan

    # For each character, store a dictionary of {word : embedding} pairs 
    for i, row in df.iterrows():
        char_embedding = {}

        # If description is NaN, skip the character
        if type(row['descriptions']) == float:
            continue

        for word in row['descriptions']:
            word = word.lower()
            # If the word was already embedded, use the embedding from the vocabulary
            if word in vocab:
                char_embedding[word] = vocab[word]

            # If it's a new word, embed it and add it to the vocabulary
            word_vector = nlp_spacy(word).vector.reshape(1, -1).astype('float32')

            # If embedding is all zeros, the word is not recognized
            if np.count_nonzero(word_vector) == 0:
                continue

            vocab[word] = word_vector
            char_embedding[word] = word_vector

        # Store the character dictionary in descriptions_embeddings
        df.at[i, 'descriptions_embeddings'] = char_embedding
    return df

def weight_embeddings(df):
    ''' Compute a weighted average of all word embeddings by weighing with 
    (1 - cosine similarity) with regards to the average vector of all characters. '''

    df['weighted_description'] = np.empty([df.shape[0], 300]).tolist()
    
    # Compute the average normalized vector of all characters
    avg_vector = np.zeros(300)

    for i, character in df.iterrows():

        embedding = character['descriptions_embeddings']

        # If no word recognized, store NaN and skip the character
        if type(embedding) == float or len(embedding) == 0:
            df.at[i, 'weighted_description'] = np.nan
            continue

        # Compute the average vector of all word embeddings of the character
        avg_word = np.zeros(300)
        for word in embedding:
            word_vector = embedding[word]
            word_vector = word_vector / np.linalg.norm(word_vector)
            avg_word += word_vector
        avg_word = avg_word / len(embedding)

        avg_vector += avg_word

    # Compute the average vector of all characters
    avg_vector = (avg_vector / len(df)).flatten()

    # For each character, weigh the embeddings by 1-cosine similarity with the average vector
    for i, character in df.iterrows():
        embedding = character['descriptions_embeddings']

        # If NaN or no words, skip the character
        if type(embedding) == float or len(embedding) == 0:
            continue
        
        # Compute the weights of all word embeddings of the character
        weights = []

        # Compute weight of each word
        for word in embedding:
            word_vector = embedding[word].flatten()
            weight = spatial.distance.cosine(word_vector, avg_vector)
            weights.append(weight)
            
        # Normalize weights to have sum = 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        # Compute the weighted average of all word embeddings of the character
        weighted_vector = np.zeros(300)
        for j, word in enumerate(embedding):

            # Normalize word vector and compute weighted average
            word_vector = embedding[word]
            word_vector = word_vector / np.linalg.norm(word_vector)
            weighted_vector = weighted_vector + word_vector * weights[j]
        
        
        # Store the weighted average in the dataframe
        df.at[i, 'weighted_description'] = weighted_vector#[0]

    return df

# --------------- Dimensionality reduction ----------------- #

def descriptions_PCA(df, n_components=3):
    ''' Apply PCA to the embeddings of the descriptions and store the results in the dataframe.'''
    
    # Remove all NaNs
    df = df.dropna(subset=['weighted_description'])

    # From the column descriptions_embeddings, get a matrix with the embeddings of each character of size n x 300
    X = np.array(df['weighted_description'].tolist())
    
    # Now apply PCA to the matrix X
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Store the results in the dataframe at idx
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    df['pca_3'] = X_pca[:, 2]
    
    return df

def descriptions_tSNE(df, n_components=3):
    ''' Apply t-SNE to the embeddings of the descriptions and store the results in the dataframe.'''
    # Remove all NaNs
    df = df.dropna(subset=['weighted_description'])
    
    # From the column descriptions_embeddings, get a matrix with the embeddings of each character of size n x 300
    X = np.array(df['weighted_description'].tolist())
    
    # Reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=n_components, random_state=0, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    # Store the results in the dataframe
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    df['tsne_3'] = X_tsne[:, 2]
    return df

# --------------- Clustering techniques ----------------- #

def cluster_descriptions(df, n_components): 
    ''' Perform clustering using gaussian mixture model, on the 3 principal components in the dataframe '''
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(df[['tsne_1', 'tsne_2', 'tsne_3']])
    labels = gmm.predict(df[['tsne_1', 'tsne_2', 'tsne_3']])
    df['labels'] = labels
    return df

# --------------- Visualization  ----------------- #

def plot_clusters_3d(df, title, x_axis='tsne_1', y_axis='tsne_2', z_axis='tsne_3'):
    ''' Plot the clusters in 3D '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_axis], df[y_axis], df[z_axis], c=df['labels'])
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    ax.set_xlim(df[x_axis].min(), df[x_axis].max())
    ax.set_ylim(df[y_axis].min(), df[y_axis].max())
    ax.set_zlim(df[z_axis].min(), df[z_axis].max())
    plt.title(title)
    plt.show()


# Remove stopwords and non-alphabetical characters from a given text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered = [w for w in word_tokens if not w.lower() in stop_words]
    #remove non-alphabetical characters (punctuation, numbers, etc.)
    filtered = [w for w in filtered if w.isalpha()]
    #stemming (reduce a word to its base form to reduce the number of unique words)
    stemmer = nltk.stem.PorterStemmer()
    filtered = [stemmer.stem(w) for w in filtered]
    #remove duplicates
    filtered = list(dict.fromkeys(filtered))
    return filtered

#extract love-related words from the summaries
def extract_love_words(text, words, threshold):
    love_words = []
    for word in words:
        #if word is empty, skip
        if word == "":
            continue
        love_words += [token.text for token in nlp_spacy(' '.join(text)) if token.similarity(word.text) > threshold]
    return love_words






