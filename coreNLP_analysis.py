''' 
Applied Data Analysis @ EPFL
Team: ToeStewBrr - Alexander Sternfeld, Marguerite Thery, Antoine Bonnet, Hugo Bordereaux 
Project: Love stories in movies
Dataset: CMU Movie Summary Corpus
'''

import os
import xml.etree.ElementTree as ET
from nltk.tree import Tree
import itertools

from load_data import *

XML_DIR = 'Data/CoreNLP/corenlp_plot_summaries_xml'
XML_DIR_ROMANCE = 'CoreNLP/RomanceOutputs'

# Given a movie ID, get the file tree from xml CoreNLP output
def get_tree(movie_id):
    xml_filename = os.path.join(XML_DIR, '{}.xml'.format(movie_id))
    tree = ET.parse(xml_filename)
    return tree

# Given a movie ID, get the file tree from xml CoreNLP augmented pipeline output
def get_tree_romance(movie_id):
    xml_filename = os.path.join(XML_DIR_ROMANCE, '{}.xml'.format(movie_id))
    tree = ET.parse(xml_filename)
    return tree

# Given an xml tree, we return all of its CoreNLP parsed sentences 
def get_parsed_sentences(tree):
    sentences = []
    for child in tree.iter():
        if child.tag == "parse":
            sentences.append(child.text)
    return sentences

# To print parsed sentences as a pretty tree. 
def print_tree(parsed_string):
    tree = Tree.fromstring(parsed_string)
    tree.pretty_print()

# Given an xml tree, get all of its characters as consecutive PERSON tags
def get_characters(tree):
    characters = []
    current_word = None
    was_person = False
    character = ''
    for child in tree.iter():
        if child.tag == 'word':
            current_word = child.text
        if child.tag == 'NER' and child.text == 'PERSON':
            if was_person:# Continue the character
                character += ' ' + current_word
            else: # Start the character
                character = current_word
                was_person = True
        if was_person and child.tag == 'NER' and child.text != 'PERSON': # End the character
            characters.append(character)
            character = ''
            was_person = False
    return characters
    
# DISCARD THIS METHOD?
# Given a character in a movie, find all sentences mentioning the character
def sentences_with_character(xml_filename, char_name):
    char_sentences = []
    if os.path.isfile(xml_filename):
        sentences = get_parsed_sentences(xml_filename)
        char_sentences = [sentence for sentence in sentences if char_name in sentence]
    return char_sentences

# Given a list of characters and a partial character name, find the full name of the character
def get_full_name(string, characters):
    ''' 
    Find the longest name of a given character in a list of character names. 
    Input: 
        string: character name (partial or full)
        characters: list of character names
    Output: 
        full_name: longest name of character found in characters
    '''
    names = string.split(' ')
    max_length = 0
    for character in characters:
        char_names = character.split(' ')
        if set(names) <= set(char_names): 
            num_names = len(char_names)
            if num_names > max_length:
                max_length = num_names
                full_name = character
    return full_name

# Helper function: given a list of characters, make a dictionary with all maps (short name : full name)
def full_name_dict(characters): 
    full_names = {}
    for character in characters:
        full_names[character] = get_full_name(character, characters)
    return full_names

# Given a list of characters names, create a dictionary of characters with 
# keys being their full name and values being the number of times they appear in the list

def aggregate_characters(characters):
    ''' 
    Input: list of characters
    Output: dictionary of (full name : number of times name is mentioned in list)
    Example: ['Harry Potter', 'Voldemort', 'Harry'] -> {'Harry Potter': 2, 'Voldemort': 1}
    '''
    character_dict = dict()
    for character in characters:
        full_character = get_full_name(character, characters)
        if full_character in character_dict:
            character_dict[full_character] += 1
        else:
            character_dict[full_character] = 1
    return character_dict

# Given a parse tree, extract the most mentioned characters in decreasing order
def most_mentioned(movie_id):
    '''
    Input: 
        tree: parse tree of the xml file
        N: the number of characters to return
    Output:
        A dictionary of the N characters most mentioned in the movie
    '''
    tree = get_tree(movie_id)
    characters = get_characters(tree)
    if len(characters) == 0:
        return None
    character_dict = aggregate_characters(characters)
    sorted_characters = sorted(character_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_characters

# Given a list of characters and a string sentence, find the sequence of characters appearing in the sentence
def characters_in_sentence(characters, sentence):
    '''
    Input: 
        characters: list of characters
        sentence: string
    Output: 
        characters_mentioned: list of characters that appear in the sentence
        characters: list of remaining characters
    Example: characters_in_sentence(['Harry Potter', 'Voldemort', 'Dumbledore'], 'Harry Potter fights Voldemort bravely.')
    Output: ['Harry Potter', 'Voldemort'], ['Dumbledore']
    '''
    characters_mentioned = []
    if (len(characters) == 0):
            return characters_mentioned, characters
    character = characters.pop(0)
    while character in sentence:
        sentence = sentence.split(character, 1)[1]
        characters_mentioned.append(character)
        if (len(characters) == 0):
            break
        character = characters.pop(0)
    return characters_mentioned, characters

# We define a method that takes in a movie ID, and outputs the number of common mentions 
# (i.e. interactions) for each pair of characters. 
def character_pairs(movie_id, plot_df):
    ''' 
    Find all pairs of characters that appear in the same sentence in a movie plot summary. 
    Input: 
        movie_id: integer Movie ID
    Output:
        sorted_pairs: A list of all character pairs in the movie in decreasing order of frequency
    '''
    char_pairs = dict()

    # Parse xml file and get all characters from plot summary
    tree = get_tree(movie_id)
    characters = get_characters(tree)
    full_name_map = full_name_dict(characters) # all maps (partial name : full name)

    # Split plot summary into sentences
    plot_summary = plot_df.loc[plot_df['Wikipedia ID'] == movie_id]['Summary'].values[0]
    sentences = re.split(r'(?<=[.!?])\s+', plot_summary)

    # For each sentence, find the characters mentioned and their full names, then add count for each pair between them
    for sentence in sentences:
        # Only consider sentences with at least 2 mentioned characters
        characters_mentioned, characters = characters_in_sentence(characters, sentence)
        if len(characters_mentioned) >= 2: 
            
            # Get full names of each character mentioned, remove doubles
            full_mentioned = set([full_name_map[c] for c in characters_mentioned])

            # Add count for each pair of characters
            pairs = list(itertools.combinations(sorted(full_mentioned), 2))
            for pair in pairs:
                if pair in char_pairs:
                    char_pairs[pair] += 1
                else:
                    char_pairs[pair] = 1
    if len(char_pairs) == 0: 
        return None
    # Sort character pairs by number of times they appear together
    sorted_pairs = sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs


# We define a method that takes in a list of movie genres and outputs their plot summaries 
def get_plots(genres, movie_df, plot_df):
    '''
    Find all movies of specified genres and return a dataframe containing their id and summaries
    Input: 
        genres: list of genres
        movie_df: dataframe containing movies' Wikipedia ID and genres 
        plot_df: dataframe containing movies' Wikipedia ID and their plot summaries 
    Output:
        genres_plots: dataframe containing movies' Wikipedia ID and their plot summaries
    '''
    is_genres = lambda i: lambda x: any(y in genres[i] for y in x) if type(x) == list else False
    movie_of_genres = movie_df[movie_df['Genres'].apply(is_genres(slice(0, len(genres))))]
    genres_plots = movie_of_genres.merge(plot_df, on='Wikipedia ID', how='left')[['Wikipedia ID', 'Summary']]
    genres_plots = genres_plots[~genres_plots['Summary'].isna()]
    return genres_plots

# Create a list of tuples containing (movie_id, subject, object) for each kbp triples with title relationship
def get_relation_df(DIR, relation_type, confidence_threshold=0.9): 
    '''
    Find all subject and object pairs that have a relation type of relation_type
    Input: 
        DIR: directory to get the xml file 
        relation_type: full list of relations can be find here https://stanfordnlp.github.io/CoreNLP/kbp.html
        confidence_threshold: float between 0 and 1, the minimum confidence of the relation
    Output:
        relations: a list of tuples (movie_id, subject, object)
    '''
    relation = []
    for filename in os.listdir(DIR):
        # Manually deleted files: 43849.xml and 1282593.xml because could not be parsed
        if filename != ".DS_Store" and filename != "43849.xml" and filename != "1282593.xml":
            movie_id = filename[:-4]
            relation.append(get_relation(movie_id, relation_type, confidence_threshold))
    # Create a dataframe with the list of tuples
    relation_df = pd.DataFrame([item for sublist in relation for item in sublist], columns=['Wikipedia ID', 'Subject', 'Relation'])
    return relation_df   

# We define a method that takes in a movie id, relation_type and confidence threshold and 
# outputs a list of tuples containing all subject and object pairs in the movie that have this type of relation
def get_relation(movie_id, relation_type, confidence_threshold=0.9):
    '''
    Find all subject and object pairs that have a relation type of relation_type
    Input: 
        movie_id: Wikipedia ID of the movie
        relation_type: full list of relations can be found here https://stanfordnlp.github.io/CoreNLP/kbp.html
        confidence_threshold: float between 0 and 1, the minimum confidence of the relation
    Output:
        relations: a list of tuples (movie_id, per_type, subject, object)
    '''
    tree = get_tree_romance(movie_id)
    relations = []
    isRelationType = False
    # Iterate through the tree
    for child in tree.iter():
        # Once at kbp section, find the triple (subject, relation, object) of the correct relation type
        if child.tag == 'kbp':
            for triple in child.iter():
                if triple.tag == 'triple':
                    # Check if confidence level is above threshold
                    confidence = float(triple.attrib['confidence'].replace(',', '.'))
                    if confidence > confidence_threshold: 
                        for element in triple.iter():
                            # Store the subject 
                            if element.tag == 'subject':
                                for el in element.iter():
                                    if el.tag == 'text':
                                        subject = el.text
                            # Store the relation 
                            if element.tag == 'relation':
                                for el in element.iter():
                                    if el.tag == 'text':
                                        if el.text == relation_type:
                                            isRelationType = True
                                            relation = el.text
                            # If the relation type is correct, store the triple
                            if element.tag == 'object' and isRelationType:
                                for el in element.iter():
                                    if el.tag == 'text':
                                        object = el.text
                                        relations.append((movie_id, relation, subject, object))
                                        isRelationType = False
    return relations


def get_per(category, store=True):
    '''
    Category should be in line with the per:... options from coreNLP. Get the relationships
    '''
    # if os path does not exist, load data and store it
    path = 'CoreNLP/' + category + '.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', index_col=0)
    else:
        df = get_relation_df(DIR='CoreNLP/RomanceOutputs',
                             relation_type='per:'+category)
        # Have list of attributes for each character
        if store:
            # Store the title_df in a csv
            df.to_csv(path, sep='\t')
    return df
