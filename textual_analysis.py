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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#Load the spacy model
nltk.download('stopwords')
nltk.download('punkt')
nlp_spacy = spacy.load("en_core_web_lg")

# --------------- Embedding ----------------- #

def construct_descriptions_embeddings(df, nlp_spacy):
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
            else:
                # If it's a new word, embed it and add it to the vocabulary
                word_vector = nlp_spacy(word).vector.reshape(
                    1, -1).astype('float32')
                # If embedding is all zeros, the word is not recognized
                if np.count_nonzero(word_vector) == 0:
                    continue
                else:
                    vocab[word] = word_vector
                    char_embedding[word] = word_vector
        # Store the character dictionary in descriptions_embeddings
        df.at[i, 'descriptions_embeddings'] = [char_embedding]
    return df

def embeddings_categorical(df):
    # Create a column 'attributes_embeddings', with the average of the embeddings corresponding to the words in the column 'attributes'
    # Column 'description_embeddings' contains a dictionary with the embeddings of each word in the description
    df = df.apply(lambda row: dict_embeddings(row, 'attributes', row['descriptions_embeddings']), axis=1)
    df = df.apply(lambda row: dict_embeddings(row, 'title', row['descriptions_embeddings']), axis=1)
    df = df.apply(lambda row: dict_embeddings(row, 'agent_verbs', row['descriptions_embeddings']), axis=1)
    df = df.apply(lambda row: dict_embeddings(row, 'patient_verbs', row['descriptions_embeddings']), axis=1)
    return df

def dict_embeddings(row, column, embeddings):
    # Create a dictionary with the embeddings of each word in the column 'column', store the dictionary in the column 'column_embeddings'
    if type(row[column]) == float:
        row[column + '_embeddings'] = np.nan
    else:
        embeddings_dic = embeddings[0]
        dict = {}
        for word in row[column]:
            if word in embeddings_dic:
                dict[word] = embeddings_dic[word]
        row[column + '_embeddings'] = dict
    return row


# ------------- Weighting embeddings --------------- #

def weight_embeddings(df, column, percentile=0, title_weight=0):
    ''' Compute a weighted average of all word embeddings of each type by weighing with 
    (1 - cosine similarity) with regards to the average vector of that type of all characters. 
    Inputs: 
        df: dataframe of word embeddings
        column: column of embeddings to weigh ('descriptions', 'agent_verbs', 'patient_verbs', 'attributes')
        percentile: to compute weighted average, only take into account words in top cos_diff percentile %
        title_weight: weight of the title in the weighted average of description, if present.
            (if 0, counts as any other word) 
    '''
    df = df.copy(deep=True)

    # If we have already weighted that column, skip. 
    embed_column = column + '_embeddings'
    newname = 'weighted_' + embed_column
    weight_column = column + '_weights'

    if newname in df.columns:
        return df 

    # If we are weighting the descriptions, first weight other columns then aggregate 
    if column == 'descriptions': 

        # Weight the other columns if not already done
        columns = ['agent_verbs', 'patient_verbs', 'attributes', 'title']
        for col in columns: 
            df = weight_embeddings(df, col, percentile, title_weight)

        # Gather number of filtered words per column
        agent_verbs_num = df['agent_verbs_weights'].apply(lambda x: len(x) if type(x) != float else 0)
        patient_verbs_num = df['patient_verbs_weights'].apply(lambda x: len(x) if type(x) != float else 0)
        attributes_num = df['attributes_weights'].apply(lambda x: len(x) if type(x) != float else 0)
        titles_num = df['title_weights'].apply(lambda x: len(x) if type(x) != float else 0)
        total_num = agent_verbs_num + patient_verbs_num + attributes_num

        # Get a list of all filtered words
        concat_words = lambda x: [] if type(x) == float else list(x[0].keys())
        df['filtered_descriptions'] = df['agent_verbs_weights'].apply(concat_words)
        df['filtered_descriptions'] += df['patient_verbs_weights'].apply(concat_words)
        df['filtered_descriptions'] += df['attributes_weights'].apply(concat_words)
        df['filtered_descriptions'] += df['title_weights'].apply(concat_words)
        df['filtered_descriptions'] = df['filtered_descriptions'].apply(lambda x: np.nan if len(x) == 0 else x)

        # Weighted average of all other columns by frequency, give weight to title if desired
        nan_to_zero = lambda x: x if type(x) != float else 0 # avoid NaN embeddings
        zero_to_one = lambda x: x if x != 0 else 1 # avoid division by zero
        if title_weight == 0: 
            total_num += titles_num
            df[newname] = df['weighted_title_embeddings'].apply(nan_to_zero) * titles_num / total_num.apply(zero_to_one)
            df[newname] += df['weighted_agent_verbs_embeddings'].apply(nan_to_zero) * agent_verbs_num / total_num.apply(zero_to_one)
            df[newname] += df['weighted_patient_verbs_embeddings'].apply(nan_to_zero) * patient_verbs_num / total_num.apply(zero_to_one)
            df[newname] += df['weighted_attributes_embeddings'].apply(nan_to_zero) * attributes_num / total_num.apply(zero_to_one)
        else:
            df[newname] = df['weighted_agent_verbs_embeddings'].apply(nan_to_zero) * agent_verbs_num / total_num.apply(zero_to_one)
            df[newname] += df['weighted_patient_verbs_embeddings'].apply(nan_to_zero) * patient_verbs_num / total_num.apply(zero_to_one)
            df[newname] += df['weighted_attributes_embeddings'].apply(nan_to_zero) * attributes_num/ total_num.apply(zero_to_one)
            df[newname] *= (1-title_weight)
            df[newname] += df['weighted_title_embeddings'].apply(nan_to_zero) * title_weight

        # Remove weight columns if we have weighted the embeddings for descriptions
        df = df.drop(columns=['agent_verbs_weights', 'patient_verbs_weights', 'attributes_weights', 'title_weights'])

        # Convert all zeros to Nans in weighted embeddings
        df[newname] = df[newname].apply(lambda x: np.nan if type(x) == float else x)

        return df

    df[newname] = np.empty([df.shape[0], 300]).tolist()

    # Compute the average normalized vector of all characters for that column
    avg_vector = np.zeros(300)
    for i, character in df.iterrows():
        embedding = character[embed_column]

        # If no word recognized, store NaN and skip the character
        if type(embedding) == float or len(embedding) == 0:
            continue

        # Compute average word embedding of that type over all characters
        avg_word = np.zeros(300)
        for word in embedding:
            word_vector = np.squeeze(embedding[word])
            avg_word += word_vector
        avg_word = avg_word / len(embedding)

        avg_vector += avg_word 

    # Compute the average vector of all characters
    avg_vector = (avg_vector / len(df)).flatten()

    # Initialize weighted average
    df[weight_column] = np.nan

    # For each character, weigh the embeddings by 1-cosine similarity with the average vector
    for i, character in df.iterrows():
        embedding = character[embed_column]

        # If NaN or no words, skip the character
        if type(embedding) == float or len(embedding) == 0:
            df.at[i, newname] = np.nan
            continue

        words = list(embedding.keys())
        
        # Compute the weights of all word embeddings of the character: 
        # weight = (cosine diff with average) * (frequency of word)
        weights = []
        for word in embedding:
            word_vector = np.squeeze(embedding[word])
            cos_diff = spatial.distance.cosine(word_vector, avg_vector)
            weight = cos_diff * words.count(word)
            weights.append(weight)

        # Only keep weights above the percentile
        weights = np.array(weights)
        min_weight = np.percentile(weights, percentile)
        weights[weights < min_weight] = 0

        # Normalize weights to have sum = 1
        norm_weights = weights / np.sum(weights)

        # Compute the weighted average of all normalized word embeddings of the character
        weight_dict = {}
        weighted_vector = np.zeros(300)
        for j, word in enumerate(embedding): 
            word_vector = np.squeeze(embedding[word]).flatten()
            norm_word_vector = word_vector / np.linalg.norm(word_vector)
            weighted_vector += norm_word_vector * norm_weights[j]
            weight_dict[word] = norm_weights[j]
        
        # Store the weighted average in the dataframe
        df.at[i, newname] = weighted_vector
        df.at[i, weight_column] = [weight_dict]

    return df

# --------------- Dimensionality reduction ----------------- #

def descriptions_PCA(df, column, n_components=3):
    ''' Apply PCA to the embeddings of the descriptions and store the results in the dataframe.'''
    
    # Remove all NaNs
    df = df.dropna(subset=[column])

    # From the column descriptions_embeddings, get a matrix with the embeddings of each character of size n x 300
    X = np.array(df[column].tolist(), dtype=object)

    # Standardize the columns of X
    X = StandardScaler().fit_transform(X)
    
    # Now apply PCA to the matrix X
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    # Store the transformed result as 'weighted_description' column in df
    if n_components > 3: 
        df[column] = X_pca.tolist()
    elif n_components == 3: 
        df['pca_1_'+column] = X_pca[:, 0]
        df['pca_2_'+column] = X_pca[:, 1]
        df['pca_3_'+column] = X_pca[:, 2]
    return df

def descriptions_tSNE(df, column, n_components=3, learning_rate='auto'):
    ''' Apply t-SNE to the embeddings of the descriptions and store the results in the dataframe.'''
    # Remove all NaNs from the column
    df = df.dropna(subset=[column])
    
    # From the column descriptions_embeddings, get a matrix with the embeddings of each character of size n x 300
    X = np.array(df[column].tolist())

    # Standardize the columns of X	
    X = StandardScaler().fit_transform(X)
    
    # Reduce the dimensionality of the embeddings
    tsne = TSNE(n_components=n_components, random_state=0, init='pca', learning_rate=learning_rate)
    X_tsne = tsne.fit_transform(X)

    # Store the results in the dataframe
    df['tsne_1_' + column] = X_tsne[:, 0]
    df['tsne_2_' + column] = X_tsne[:, 1]
    df['tsne_3_' + column] = X_tsne[:, 2]
    return df

# --------------- Clustering techniques ----------------- #

def refine_clusters(df, column, type, rate=0.2):
    ''' Remove outliers from the clusters.
    Input:
    - rate: rate of points to remove from each cluster
    '''
    x_axis = type+'_1_'+column
    y_axis = type+'_2_'+column
    z_axis = type+'_3_'+column
    for label in df['labels'].unique():
        mean_point = df.loc[df['labels'] == label,
                            [x_axis, y_axis, z_axis]].mean()
        # Calculate distance of each point to the mean point
        df.loc[df['labels'] == label, 'distance'] = np.sqrt(np.sum(
            (df.loc[df['labels'] == label, [x_axis, y_axis, z_axis]] - mean_point)**2, axis=1))
        # Calculate the number of points to remove according to the rate parameter
        nbr_points_cluster = len(df.loc[df['labels'] == label])
        nbr_points_to_remove = int(nbr_points_cluster * rate)
        # Sort the points by descending distance and remove the nbr_points_to_remove with the highest distance
        ind_points_to_remove = df.loc[df['labels'] == label].sort_values(
            by='distance', ascending=False).iloc[nbr_points_to_remove:].index
        df = df.drop(ind_points_to_remove)
    df = df.drop(columns=['distance'])
    return df

def GMM_cluster(df, column, n_components, method='tsne'): 
    ''' Perform clustering using gaussian mixture model, on the 3 principal components in the dataframe '''
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    dim1 = method + '_1_' + column
    dim2 = method + '_2_' + column
    dim3 = method + '_3_' + column
    gmm.fit(df[[dim1, dim2, dim3]])
    labels = gmm.predict(df[[dim1, dim2, dim3]])
    df['labels'] = labels
    return df

def DBSCAN_cluster(df, column, method='tsne', eps=5, min_samples=50): 
    ''' Perform clustering using DBSCAN, on the 3 principal components in the dataframe 
    Inputs: 
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
    '''
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    dim1 = method + '_1_' + column
    dim2 = method + '_2_' + column
    dim3 = method + '_3_' + column

    db = dbscan.fit(df[[dim1, dim2, dim3]])
    labels = db.labels_
    df['labels'] = labels

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    # Remove noise points
    df = df[df['labels'] != -1]

    return df, n_clusters, n_noise

# --------------- Visualization  ----------------- #

def plot_clusters_3d(df, title, column, method='tsne'):
    ''' Plot the clusters in 3D '''
    x_axis = method + '_1_' + column
    y_axis = method + '_2_' + column
    z_axis = method + '_3_' + column
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_axis], df[y_axis], df[z_axis], c=df['labels'], cmap=plt.get_cmap('tab20'))
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    #ax.set_xlim(df[x_axis].min(), df[x_axis].max())
    #ax.set_ylim(df[y_axis].min(), df[y_axis].max())
    #ax.set_zlim(df[z_axis].min(), df[z_axis].max())
    plt.title(title)
    plt.show()


# --------------- Main ----------------- #

def cluster_embeddings(df, desc='descriptions', min_words=0, eps=5, min_samples=50, percentile=50, title_weight=0.3):
    ''' Experiment with embeddings. PCA -> t-SNE -> DBSCAN clustering
    Inputs: 
        df: dataframe with weighted categorical embeddings
        desc: type of description (title, attributes, descriptions) to use 
        sample: sample of the data to use (1: all data) (for t-SNE speedup)
        min_words: filter out characters with less than min_words descriptive words
        DBSCAN parameters: eps, min_samples
    '''
    df = df.copy(deep=True)

    # Ablation if desired
    if min_words > 0:
        df = df[df[desc].apply(lambda x: len(x) if type(x) == list else 0) >= min_words]

    # Weighing or averaging 
    column = 'weighted_' + desc + '_embeddings'

    # Dimensionality reduction: PCA to 50 dimensions -> t-SNE to 3 dimensions
    n_total = df[column].apply(lambda x: 1 if type(x) == np.ndarray else 0).sum()
    df = descriptions_PCA(df, column=column, n_components=50)
    df = descriptions_tSNE(df, column=column, n_components=3, learning_rate='auto')

    # DBSCAN Clustering
    df, n_clusters, n_removed = DBSCAN_cluster(df, column, method='tsne', eps=eps, min_samples=min_samples)
    title = 't-SNE + DBSCAN with {} clusters, \nRemoved {}/{} noisy data points\nDBSCAN: eps = {}, min_samples = {}\nWeighing: weighing = {}, percentile = {}, title_weight = {}\nFilter: desc = {}, sample={}, min_words={}'.format(
        n_clusters, n_removed, n_total, eps, min_samples, weighing, percentile, title_weight, desc, sample, min_words)
    
    # Plot clusters
    plot_clusters_3d(df, title, column=column)

    return df

# --------------- Cluster analysis ----------------- #


def filter_descriptions(cluster_df):
    # Apply np.squeeze on every row of the column descriptions_embeddings
    cluster_df['descriptions_embeddings'] = cluster_df['descriptions_embeddings'].apply(
        lambda x: np.squeeze(x) if type(x) == list else x)

    # Apply np.squeeze on each value in the dictionary of the column descriptions_embeddings
    cluster_df['descriptions_embeddings'] = cluster_df['descriptions_embeddings'].apply(
        lambda x: {key: np.squeeze(x[key]) for key in x} if type(x) == dict else x)

    # Find the average embedding over all rows in descriptions_embeddings, each row is a dictionary.
    # The dictionary has as keys the words and as values the embedding.
    avg_descr = np.mean([np.mean(list(x.values()), axis=0)
                        for x in cluster_df['descriptions_embeddings'].values if type(x) == dict], axis=0)
    # import cosine_similarity from sklearn.metrics.pairwise
    from sklearn.metrics.pairwise import cosine_similarity
    # Column descriptions_embeddings contains a dictionary with as keys the words and as values a list with the embedding as the first element.
    # Get the three keys of which the values have the highest cosine similarity to avg_descr
    cluster_df['filtered_descriptions'] = cluster_df['descriptions_embeddings'].apply(
        lambda x: sorted(x, key=lambda word: cosine_similarity(x[word].reshape(300, -1), avg_descr.reshape(300, -1))[0][0], reverse=True)[:3] if type(x) == dict else x)
    return cluster_df


# --------------- TO REVIEW  ----------------- #


# Remove stopwords and non-alphabetical characters from a given text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    filtered = [w for w in word_tokens if not w.lower() in stop_words]
    filtered = [w for w in filtered if w.isalpha()]
    stemmer = nltk.stem.PorterStemmer()
    filtered = [stemmer.stem(w) for w in filtered]
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









