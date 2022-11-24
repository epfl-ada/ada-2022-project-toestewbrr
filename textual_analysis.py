''' 
Applied Data Analysis @ EPFL
Team: ToeStewBrr - Alexander Sternfeld, Marguerite Thery, Antoine Bonnet, Hugo Bordereaux 
Project: Love stories in movies
Dataset: CMU Movie Summary Corpus
'''

import os
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import string
import re


nltk.download('stopwords')
nltk.download('punkt')

#Load the spacy model
nlp_spacy = spacy.load("en_core_web_lg")

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






