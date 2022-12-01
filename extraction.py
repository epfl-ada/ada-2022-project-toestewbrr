import pandas as pd 
from coreNLP_analysis import *

AGENT_VERBS = ['nsubj', 'obl:agent']
PATIENT_VERBS = ['nsubj:pass', 'nsubj:xsubj', 'obj']
ATTRIBUTE_TYPES = ['appos', 'amod', 'nmod:poss', 'nmod:of']
TAGS = ['per:spouse', 'per:title', 'per:religion', 'per:age']
CORENLP_OUTPUT_DIR = 'Data/CoreNLP/PlotsOutputs'
THRESHOLD = 0.8


# -------------------- Extracting depparse annotations and KBP relationships --------------------#

def parse_description_relation(tree, relation_types, tags, threshold): 
    ''' Given a xml file parsed into a tree, extracts: 
        - all depparse annotations (subject, object, relation_type) 
        - all KBP relations (subject, object, relation_type).  '''
    description = []
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
            if type in relation_types:
                description.append((governor, dependent, type))
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
    return movie_id, description, relations



# -------------------- Filter characters --------------------#

def filter_description(descriptions, full_names):
    '''
    Convert partial names in depparse relations to full names, keep only those in the character list.
    Input: 
        relations: list of tuples (partial_name, attribute, relation_type)
        full_names: full-name dictionary mapping {partial_name: full:name}
    Output:
        filter_pairs: list of tuples (full_name, attribute, relation_type)
    '''
    filter_descriptions = []
    for pair in descriptions: 
        try: 
            full_name1 = full_names[pair[0]]
        except KeyError: 
            full_name1 = None
        try:
            full_name2 = full_names[pair[1]]
        except KeyError:
            full_name2 = None
        if full_name1 is not None and full_name2 is None: 
            filter_descriptions.append((full_name1, pair[1], pair[2]))
        elif full_name1 is None and full_name2 is not None:
            filter_descriptions.append((full_name2, pair[0], pair[2]))

    return filter_descriptions

def filter_relations(relations, full_names):
    '''
    Convert partial names in KBP relations to full names, keep only those in the character list.
    Input: 
        relations: list of tuples (subject, object, relation_type)
        full_names: full-name dictionary mapping {partial_name: full:name}
    Output:
        filter_relations: list of tuples (full_name, object, relation_type)
    '''
    filter_relations = []
    for rel in relations:
        try: 
            subject = full_names[rel[0]]
        except KeyError: 
            subject = None
        try:
            object = full_names[rel[1]]
        except KeyError:
            object = None
        # Remove self-love relationships
        if subject is not None and object is not None and subject != object: 
            filter_relations.append((subject, object, rel[2]))

    return filter_relations

# -------------------- Creating dataframes --------------------#


def split_description(descriptions): 
    ''' Split the description pairs by relation types: {agent verb, patient verb, attribute}.'''
    agent_verbs = []
    patient_verbs = []
    attributes = []

    for pair in descriptions: 
        if pair[2] in AGENT_VERBS:
            agent_verbs.append(pair[:2])
        elif pair[2] in PATIENT_VERBS:
            patient_verbs.append(pair[:2])
        elif pair[2] in ATTRIBUTE_TYPES:
            attributes.append(pair[:2])

    agent_verbs = pd.DataFrame(agent_verbs, columns=['character', 'agent_verb'])
    patient_verbs = pd.DataFrame(patient_verbs, columns=['character', 'patient_verb'])
    attributes = pd.DataFrame(attributes, columns=['character', 'attribute'])
    return agent_verbs, patient_verbs, attributes

def split_relations(relations):
    ''' Split the relations by relation types: spouse, title, religion, age.'''
    spouse = []
    title = []
    religion = []
    age = []

    for rel in relations:
        if rel[2] == TAGS[0]:
            spouse.append(rel[:2])
        elif rel[2] == TAGS[1]:
            title.append(rel[:2])
        elif rel[2] == TAGS[2]:
            religion.append(rel[:2])
        elif rel[2] == TAGS[3]:
            age.append(rel[:2])

    spouse = pd.DataFrame(spouse, columns=['subject', 'object'])
    title = pd.DataFrame(title, columns=['character', 'title'])
    religion = pd.DataFrame(religion, columns=['character', 'religion'])
    age = pd.DataFrame(age, columns=['character', 'age'])
    return spouse, title, religion, age


def get_descriptions_relations_helper(movie_id, descriptions, relations):
    ''' Create dataframes for attributes and relations for a given movie.
    Output: 
        descriptions_df: dataframe with columns (movie_id, character, attribute)
        relations_df: dataframe with columns (movie_id, subject, object)
    '''
    # Split the descriptions by relation types: {agent verb, patient verb, attribute}
    agent_verbs, patient_verbs, attributes = split_description(descriptions)
    
    # Split the relations by relation types: {spouse, title, religion, age}
    spouse, title, religion, age = split_relations(relations)

    # Create a dataframe containing all the info for each character
    descriptions_df = pd.merge(agent_verbs, patient_verbs, attributes, title, religion, age, how='outer', on='character')
    
    # Create dataframe containing all the love relationships between characters
    relations_df = pd.DataFrame(spouse, columns=['subject', 'object'])

    # Add movie_id column to each dataframe
    descriptions_df['movie_id'] = movie_id
    relations_df['movie_id'] = movie_id

    return descriptions_df, relations_df


# -------------------- Extraction of a single movie --------------------#

def get_descriptions_relations(tree): 
    ''' Given a xml parsed tree and depparse annotation pairs, extracts character descriptions & relations.'''

    # Get full name of each character
    characters = get_characters(tree)
    full_names = full_name_dict(characters)

    # Extract depparse pairs and KBP relations
    relation_types = AGENT_VERBS + PATIENT_VERBS + ATTRIBUTE_TYPES
    movie_id, descriptions, relations = parse_description_relation(tree, relation_types, TAGS, THRESHOLD)
    
    # Remove duplicates
    descriptions = list(set(descriptions))
    relations = list(set(relations))

    # Expand character names to full names, filter out characters not in the list
    descriptions = filter_description(descriptions, full_names)
    relations = filter_relations(relations, full_names)
    
    # Create dataframes for character descriptions and relations for a given movie
    descriptions_df, relations_df = get_descriptions_relations_helper(movie_id, descriptions, relations)
 
    return descriptions_df, relations_df


# -------------------- Extraction of all movies --------------------#


def extract_descriptions_relations(log_interval = 1000): 
    ''' Given a directory of xml files, extract all character descriptions and relations and store them into dataframes. '''
    print('Extracting character descriptions & relations...')

    # Get all xml files in the directory
    xml_files = [f for f in os.listdir(CORENLP_OUTPUT_DIR) if f.endswith('.xml')]
    num_files = len(xml_files)
    
    # Create dataframes to store the extracted descriptions and relations
    descriptions = pd.DataFrame(columns=['Wikipedia ID', 'Character name', 'Agent verbs', 'Patient verbs', 'Attributes'])
    relations = pd.DataFrame(columns=['Wikipedia ID', 'Subject', 'Object', 'Relation'])

    for idx, xml_file in enumerate(xml_files):

        # Print progress every log_interval files
        if idx % log_interval == 0:
            print('\Progress: {}/{} ({}%)'.format(idx, num_files, round(idx/num_files*100, 2)))

        # Parse the xml file and extract description & relations dataframes
        tree = ET.parse(os.path.join(CORENLP_OUTPUT_DIR, xml_file))
        descriptions_df, relations_df = get_descriptions_relations(tree, THRESHOLD)

        # Concatenate descriptions to previous descriptions
        descriptions = pd.concat([descriptions, descriptions_df], ignore_index=True)
        relations = pd.concat([relations, relations_df], ignore_index=True)

    # Save descriptions to a csv file
    descriptions.to_csv('Data/CoreNLP/descriptions.csv', sep='\t')
    relations.to_csv('Data/CoreNLP/relations.csv', sep='\t')

    return descriptions, relations

