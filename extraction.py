import pandas as pd 
from coreNLP_analysis import *

AGENT_VERBS = ['nsubj', 'obl:agent']
PATIENT_VERBS = ['nsubj:pass', 'nsubj:xsubj', 'obj']
ATTRIBUTE_TYPES = ['appos', 'amod', 'nmod:poss', 'nmod:of']
TAGS = ['per:spouse', 'per:title', 'per:religion', 'per:age']

# -------------------- Extracting depparse annotations and KBP relationships --------------------#

def extract_attributes_relations(tree, attribute_types, tags, threshold): 
    ''' Given a xml file parsed into a tree, extracts all depparse annotations (subject, object, relation_type) 
    among the given relation types. '''
    pairs = []
    relations = []
    isTag = False
    for child in tree.iter():
        if child.tag == 'docId':
            movie_id = child.text[:-4] 
        if child.tag == 'dep':
            type = child.attrib['type']
        if child.tag == 'governor': 
            governor = child.text
        if child.tag == 'dependent':
            dependent = child.text
            if type in attribute_types:
                pairs.append((governor, dependent, type))
        if child.tag == 'kbp':
            for triple in child.iter():
                if triple.tag == 'triple':
                    confidence_level = float(triple.attrib['confidence'].replace(',', '.'))
                    if confidence_level > threshold:
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
                                            isTag = True
                                            tag = el.text
                            # If the relation type is correct, store the triple
                            if element.tag == 'object' and isTag:
                                for el in element.iter():
                                    if el.tag == 'text':
                                        object = el.text
                                        relations.append((subject, object, tag))
                                        isTag= False
    return movie_id, pairs, relations

# -------------------- Filter on characters --------------------#

def filtered_pairs(pairs, full_names):
        # Find all pairs containing a single character name, and add it to a list of tuples (full name, attribute, attribute type)
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

    return filtered_pairs

def filtered_relations(relations, full_names):
    filtered_relations = []
    # TODO: Make sure full names work correctly 
    for rel in relations:
        try: 
            subject = full_names[rel[0]]
        except KeyError: 
            subject = None
        try:
            object = full_names[rel[1]]
        except KeyError:
            object = None
        if subject is not None and object is not None and subject != object: 
            filtered_relations.append((subject, object, rel[2]))
        else: 
            continue

    return filtered_relations

# -------------------- Creating dataframes --------------------#

def get_attributes_relations_helper(movie_id, pairs, relations):
    # Split the pairs into relation types: agent verb, patient verb, attribute
    agent = pd.Dataframe([pair for pair in pairs if pair[2] in AGENT_VERBS], columns=['character', 'agent', 'agent_type'])
    patient = pd.DataFrame([pair for pair in pairs if pair[2] in PATIENT_VERBS], columns=['character', 'patient', 'patient_type'])
    attribute = pd.DataFrame([pair for pair in pairs if pair[2] in ATTRIBUTE_TYPES], columns=['character', 'attribute', 'attribute_type'])

    # Split the relations into relation types: spouse, title, religion, age
    spouse = pd.DataFrame([rel for rel in relations if rel[2] == TAGS[0]], columns=['subject', 'object', 'relation'])
    title = pd.DataFrame([rel for rel in relations if rel[2] == TAGS[1]], columns=['character', 'title', 'title_type'])
    religion = pd.DataFrame([rel for rel in relations if rel[2] == TAGS[2]], columns=['character', 'religion', 'religion_type'])
    age = pd.DataFrame([rel for rel in relations if rel[2] == TAGS[3]], columns=['character', 'age', 'age_type'])

    # Create a dataframe containing all the info for each character
    attributes_df = pd.merge(agent, patient, attribute, title, religion, age, how='outer', on='character')
    attributes_df.drop(['agent_type', 'patient_type', 'attribute_type', 'title_type', 'religion_type', 'age_type'], axis=1)
    
    # Create dataframe containing all the love relationships between characters
    relations_df = pd.DataFrame(spouse, columns=['subject', 'object', 'relation'])
    relations_df.drop(columns=['relation'])

    # Add movie_id column to each dataframe
    attributes_df['movie_id'] = movie_id
    relations_df['movie_id'] = movie_id

    return attributes_df, relations_df

# -------------------- Main extraction function --------------------#

def get_attributes_relations(tree, threshold_confidence_level): 
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

    # Extract depparse pairs and KBP relations
    attribute_types = AGENT_VERBS + PATIENT_VERBS + ATTRIBUTE_TYPES
    movie_id, pairs, relations = extract_attributes_relations(tree, attribute_types, TAGS, threshold_confidence_level)
    
    # Remove duplicates
    pairs = list(set(pairs))
    relations = list(set(relations))

    # Filter pairs and relations to only include the ones where the subject is a character of the movie
    pairs = filtered_pairs(pairs, full_names)
    relations = filtered_relations(relations, full_names)
    
    # Create two dataframes containing respectively all the attributes for one character and all the love relationships between characters
    attributes_df, relations_df = get_attributes_relations_helper(movie_id, pairs, relations)
 
    return attributes_df, relations_df


