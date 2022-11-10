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
def character_pairs(movie_id):
    ''' 
    Find all pairs of characters that appear in the same sentence in a movie plot summary. 
    Input: 
        movie_id: integer Movie ID
    Output:
        A list of all character pairs in the movie in decreasing order of frequency
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
    
    # Sort character pairs by number of times they appear together
    sorted_pairs = sorted(char_pairs.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs