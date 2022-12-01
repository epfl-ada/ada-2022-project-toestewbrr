''' 
Applied Data Analysis @ EPFL
Team: ToeStewBrr - Alexander Sternfeld, Marguerite Thery, Antoine Bonnet, Hugo Bordereaux 
Project: Love stories in movies
Dataset: CMU Movie Summary Corpus

This file contains the code for the coreNLP analysis of the movie summaries.
'''


import os
from  zipfile import ZipFile
import xml.etree.ElementTree as ET
from nltk.tree import Tree
import itertools

from load_data import *

XML_DIR = 'Data/CoreNLP/corenlp_plot_summaries_xml'

AGENT_VERBS = ['nsubj', 'obl:agent']
PATIENT_VERBS = ['nsubj:pass', 'nsubj:xsubj', 'obj']
ATTRIBUTE_TYPES = ['appos', 'amod', 'nmod:poss', 'nmod:of']


# ----------------- Filtering movie plot summaries  ----------------- #


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


# -------------------- Parsing coreNLP analysis output --------------------#


# Given a movie ID, get the file tree from xml CoreNLP output
def get_tree(movie_id):
    xml_filename = os.path.join(XML_DIR, '{}.xml'.format(movie_id))
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


# -------------------- Extracting characters --------------------#


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

# TO DISCARD
# Find the number of common mentions (i.e. interactions) for each pair of characters in a given movie.
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


# -------------------- Extracting KBP relationships --------------------#


# We define a method that takes in a tree, a movie_id and a list of tags.   
# It outputs a list of tuples containing all subject and object pairs in the movie plot summary which have a KBP relation of a type in tags
def extract_relations(tree, movie_id, tags):
    '''
    Find all subject and object pairs that have a relation type of relation_type
    Input: 
        tree: xml parse tree
        movie_id: Wikipedia ID of the movie
        tags: list of relations that we want to extract
    Output:
        relations: a list of tuples (movie_id, subject, object, tag, confidence_level)
    '''
    relations = []
    isRelationType = False
    # Iterate through the tree
    for child in tree.iter():
        # Once at kbp section, find the triple (subject, relation, object) of the correct relation type
        if child.tag == 'kbp':
            for triple in child.iter():
                if triple.tag == 'triple':
                    confidence_level = float(triple.attrib['confidence'].replace(',', '.'))
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
                                    if el.text in tags:
                                        isRelationType = True
                                        tag = el.text
                        # If the relation type is correct, store the triple
                        if element.tag == 'object' and isRelationType:
                            for el in element.iter():
                                if el.tag == 'text':
                                    object = el.text
                                    relations.append((movie_id, subject, object, tag, confidence_level))
                                    isRelationType = False
    return relations

# This method takes in a zip_file containing xml files of movies plot summaries processed by the CoreNLP pipeline and a list of tags. 
# It returns a dataframe of all the KBP relationships with a type in tags find in the plot summaries. 
def get_relations(path, tags): 
    '''
    Find all subject and object pairs that have a relation type in tags
    Input: 
        file: zip file containing all the plot summaries xml files 
        tags: list of relations that we want to extract, full list of relations can be find here https://stanfordnlp.github.io/CoreNLP/kbp.html
    Output:
        relations: a list of tuples (movie_id, subject, object, tag, confidence_level)
    '''
    relations = []
    with ZipFile(path, 'r') as zip:
        for filename in zip.namelist():
        # Manually deleted files: 43849.xml and 1282593.xml because could not be parsed
            movie_id = filename[:-4] # remove .xml
            with zip.open('{}.xml'.format(movie_id)) as f:
                tree = ET.parse(f)
                root = tree.getroot()  
            relations.append(extract_relations(root, movie_id, tags))
    # Create a dataframe with the list of tuples
    relations_df = pd.DataFrame([item for sublist in relations for item in sublist], columns=['Wikipedia ID', 'Subject', 'Object', 'Tag', 'Confidence Level'])
    return relations_df   

def get_per(tag):
    '''
    Category should be in line with the per:... options from coreNLP. Get the relationships
    '''
    # if os path does not exist, load data and store it
    path = 'CoreNLP/' + tag + '.csv'
    if os.path.exists(path):
        df = pd.read_csv(path, sep='\t', index_col=0)
    else: 
        print("The dataframe does not exist already. We will create it by filtering on the relations csv file.")
        df = pd.read_csv('CoreNLP/relations.csv', sep='\t', index_col=0)
        df = df[df['Tag'] == tag]
        df.to_csv(path, sep='\t')
    return df


# -------------------- Extracting depparse annotations --------------------#


def get_attributes(tree, relation_types): 
    ''' Given a xml file parsed into a tree, extracts all depparse annotations (subject, object, relation_type) 
    among the given relation types. '''
    pairs = []
    for child in tree.iter():
        if child.tag == 'dep':
            type = child.attrib['type']
        if child.tag == 'governor': 
            governor = child.text
        if child.tag == 'dependent':
            dependent = child.text
            if type in relation_types:
                pairs.append((governor, dependent, type))
    return pairs

def extract_attributes(tree): 
    ''' Given a xml parsed tree and depparse annotation pairs, extracts relations of given type, 
    removes duplicates, extracts the ones involving a character. 
    Input: 
        tree: tree to extract relations from
        relation_types: list of relation types to extract, e.g. ['nsubj', 'obj']
    Output:
        agent_pairs, patient_pairs, attribute_pairs: list of relation tuples (subject, object) of given type
    '''
    # Get full name of each character
    characters = get_characters(tree)
    full_names = full_name_dict(characters)

    # Extract depparse pairs
    relation_types = AGENT_VERBS + PATIENT_VERBS + ATTRIBUTE_TYPES
    pairs = get_attributes(tree, relation_types)

    # Remove duplicates
    pairs = list(set(pairs))

    # Find all pairs containing a single character name, and add it to a list of tuples (full name, attribute, relation type)
    filtered_pairs = []
    for pair in pairs: 
        try: 
            full_name1 = full_names[pair[0]]
        except KeyError: 
            full_name1 = None
        try:
            full_name2 = full_names[pair[1]]
        except KeyError:
            full_name2 = None
        if full_name1 is not None and full_name2 is None: 
            filtered_pairs.append((full_name1, pair[1], pair[2]))
        elif full_name1 is None and full_name2 is not None:
            filtered_pairs.append((full_name2, pair[0], pair[2]))
    
    # Split the pairs into relation types: agent verb, patient verb, attribute
    agent_pairs = []
    patient_pairs = []
    attribute_pairs = []
    for pair in filtered_pairs:
        if pair[2] in AGENT_VERBS:
            agent_pairs.append(pair[0:2])
        elif pair[2] in PATIENT_VERBS:
            patient_pairs.append(pair[0:2])
        elif pair[2] in ATTRIBUTE_TYPES:
            attribute_pairs.append(pair[0:2])
    
    return agent_pairs, patient_pairs, attribute_pairs
    
# For each character, aggregate all related actions and attributes
def get_character_description(tree):
    agent_pairs, patient_pairs, attribute_pairs = extract_attributes(tree)
    
    attributes = {}
    patient_verbs = {}
    agent_verbs = {}

    for (char, verb) in agent_pairs:
        if char not in agent_verbs:
            agent_verbs[char] = [verb]
        else:
            agent_verbs[char].append(verb)

    for (char, verb) in patient_pairs:
        if char not in patient_verbs:
            patient_verbs[char] = [verb]
        else:
            patient_verbs[char].append(verb)

    for (char, attr) in attribute_pairs:
        if char not in attributes:
            attributes[char] = [attr]
        else:
            attributes[char].append(attr)
    
    return agent_verbs, patient_verbs, attributes

# -------------------- Main extraction function --------------------#

