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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt


#Load the spacy model
nltk.download('stopwords')
nltk.download('punkt')
nlp_spacy = spacy.load("en_core_web_lg")

# --------------- Embedding ----------------- #

def embed_descriptions_avg(char_description, nlp_spacy):
    ''' Embed the description of a character using the average of the word embeddings.'''
    embeddings = np.zeros(300)
    for word in char_description:
        if word in nlp_spacy.vocab:
            embeddings = embeddings + nlp_spacy(word).vector.reshape(1, -1)
    embeddings = embeddings / len(char_description)
    embeddings = embeddings.astype('float32')
    return embeddings

def embed_descriptions(char_description, nlp_spacy):
    ''' Embed the description of a character as an array with rows as word embeddings.'''
    words_in_vocab = [word for word in char_description if word in nlp_spacy.vocab]
    char_embedding = np.array((len(words_in_vocab), 300))
    for i, word in enumerate(words_in_vocab):
        embedding = nlp_spacy(word).vector.reshape(1, -1)
        char_embedding[i] = embedding.astype('float32')
    return char_embedding

def weigh_vectors(char_embedding, nlp_spacy=nlp_spacy):
    ''' Given a (n x 300) character embedding, weigh the row vectors 
    by cosine similarity with the average vector'''
    # Compute average vector
    avg_vector = np.mean(char_embedding, axis=0)

    # Compute cosine similarity between each word vector and the average vector
    similarities = [nlp_spacy.similarity(word, avg_vector) for word in char_embedding]

    # Normalize weights
    weights = similarities / np.sum(similarities)

    # Return weighted average of the vectors
    return np.average(char_embedding, axis=0, weights=weights)


# --------------- Dimensionality reduction ----------------- #

def descriptions_PCA(df, n_components=3):
    ''' Apply PCA to the embeddings of the descriptions and store the results in the dataframe.'''
    # From the column descriptions_embeddings, get a matrix with the embeddings of each character of size n x 300
    X = np.array(df['descriptions_embeddings'].tolist())
    X = X.reshape(X.shape[0], X.shape[2])

    # Now apply PCA to the matrix X
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)

    # Store the results in the dataframe
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    df['pca_3'] = X_pca[:, 2]
    
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

def plot_clusters_3d(df, title):
    ''' Plot the clusters in 3D '''
    n_clusters = len(df['labels'].unique())
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['tsne_1'], df['tsne_2'], df['tsne_3'], c=df['labels'])
    ax.set_xlabel('tsne_1')
    ax.set_ylabel('tsne_2')
    ax.set_zlabel('tsne_3')
    ax.set_xlim(df['tsne_1'].min(), df['tsne_1'].max())
    ax.set_ylim(df['tsne_2'].min(), df['tsne_2'].max())
    ax.set_zlim(df['tsne_3'].min(), df['tsne_3'].max())
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






