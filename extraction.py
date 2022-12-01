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
    filtered_relations = []
    for rel in relations:
        try: 
            full_name1 = full_names[rel[0]]
        except KeyError: 
            full_name1 = None
        try:
            full_name2 = full_names[rel[1]]
        except KeyError:
            full_name2 = None
        # If we have two character names with per:spouse relation (remove self-love)
        if full_name1 is not None and full_name2 is not None and rel[2] == 'per:spouse' and full_name1 != full_name2:
            filtered_relations.append((full_name1, full_name2, rel[2]))
        
        # If not per:spouse and only one character name, store relation
        elif full_name1 is not None and full_name2 is None and rel[2] != 'per:spouse':
            filtered_relations.append((full_name1, rel[1], rel[2])) 
        elif full_name1 is None and full_name2 is not None and rel[2] != 'per:spouse':
            filtered_relations.append((full_name2, rel[0], rel[2]))

        
    return filtered_relations


# -------------------- Creating dataframes --------------------#


def split_description(description_pairs): 
    ''' Split the description pairs by relation types: {agent verb, patient verb, attribute}.'''
    agent_verbs = []
    patient_verbs = []
    attributes = []
    
    for pair in description_pairs: 
        if pair[2] in AGENT_VERBS:
            agent_verbs.append(pair[:2])
        elif pair[2] in PATIENT_VERBS:
            patient_verbs.append(pair[:2])
        elif pair[2] in ATTRIBUTE_TYPES:
            attributes.append(pair[:2])

    return agent_verbs, patient_verbs, attributes

def split_relations(relation_pairs):
    ''' Split the relations by relation types: spouse, title, religion, age.
    Input: 
        relation_pairs: list of tuples (subject, object, relation_type)
    Output: 
        spouses: DataFrame (subject, object)
        titles: DataFrame (character, title)
        religions: DataFrame (character, religion)
        ages: DataFrame (character, age)
    '''
    spouse = []
    title = []
    religion = []
    age = []

    for rel in relation_pairs:
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

# Extract all descriptions of each character in a single movie
def get_character_description(descriptions_pairs):
    ''' Given a xml file parsed into a tree, extracts all depparse annotations (subject, object, relation_type)
    and store them in a dictionary of dictionaries: 
        {character_full_name : {agent_verbs : [...], patient_verbs : [...], attributes : [...]}}
    '''
    descriptions_dict = {}
    agent_verbs, patient_verbs, attributes = split_description(descriptions_pairs)    
    for (char, verb) in agent_verbs:
        if char not in descriptions_dict:
            descriptions_dict[char] = {'agent_verbs': [verb], 'patient_verbs': [], 'attributes': []}
        else:
            descriptions_dict[char]['agent_verbs'].append(verb)

    for (char, verb) in patient_verbs:
        if char not in descriptions_dict:
            descriptions_dict[char] = {'agent_verbs': [], 'patient_verbs': [verb], 'attributes': []}
        else:
            descriptions_dict[char]['patient_verbs'].append(verb)

    for (char, attr) in attributes:
        if char not in descriptions_dict:
            descriptions_dict[char] = {'agent_verbs': [], 'patient_verbs': [], 'attributes': [attr]}
        else:
            descriptions_dict[char]['attributes'].append(attr)
    return descriptions_dict

def create_descriptions_relations_df(movie_id, description_pairs, relation_pairs):
    ''' Create dataframes for attributes and relations for a given movie.
    Input: 
        movie_id: movie id
        descriptions: list of tuples (full_name, attribute, attribute_type)
        relations: list of tuples (subject, object, relation_type)
    Output: 
        descriptions_df: dataframe with columns (movie_id, character, attribute)
        relations_df: dataframe with columns (movie_id, subject, object)
    '''
    # Split the descriptions by relation types: {agent verb, patient verb, attribute} for each character
    descriptions_dict = get_character_description(description_pairs)

    # Create descriptions dataframe with one row per character 
    descriptions_df = pd.DataFrame(columns=['movie_id', 'character', 'agent_verbs', 'patient_verbs', 'attributes'])
    for name, description in descriptions_dict.items():
        # Convert all empty lists to NaN values
        for key, value in description.items():
            if value == []:
                description[key] = np.nan
        # Append row to dataframe
        descriptions_df.loc[len(descriptions_df)] = [
            movie_id, name, description['agent_verbs'], description['patient_verbs'], description['attributes']
            ]

    # Split the relations by relation types: {spouse, title, religion, age}
    spouse, title, religion, age = split_relations(relation_pairs)

    # Create a dataframe containing all the info for each character
    descriptions_df = pd.merge(descriptions_df, title, how='outer', on='character')
    descriptions_df = pd.merge(descriptions_df, religion, how='outer', on='character')
    descriptions_df = pd.merge(descriptions_df, age, how='outer', on='character')

    # Create dataframe containing all the love relationships between characters
    relations_df = pd.DataFrame(spouse, columns=['subject', 'object'])

    # Add movie_id column to each dataframe
    descriptions_df['movie_id'] = movie_id
    relations_df['movie_id'] = movie_id

    return descriptions_df, relations_df


# -------------------- Extraction of a single movie --------------------#

def get_descriptions_relations(tree): 
    ''' Extracts descriptions & relations for a single movie.'''

    # Get full name of each character
    characters = get_characters(tree)
    full_names = full_name_dict(characters)

    # Extract depparse pairs and KBP relations
    relation_types = AGENT_VERBS + PATIENT_VERBS + ATTRIBUTE_TYPES
    movie_id, description_pairs, relation_pairs = parse_description_relation(tree, relation_types, TAGS, THRESHOLD)

    # Expand character names to full names, filter out characters not in the list
    filtered_descriptions = filter_description(description_pairs, full_names)
    filtered_relations = filter_relations(relation_pairs, full_names)

    # Remove duplicates
    filtered_descriptions = list(set(filtered_descriptions))
    filtered_relations = list(set(filtered_relations))
    
    # Create dataframes for character descriptions and relations for a given movie
    descriptions_df, relations_df = create_descriptions_relations_df(movie_id, filtered_descriptions, filtered_relations)

    return descriptions_df, relations_df


# -------------------- Extraction of all movies --------------------#

def extract_descriptions_relations(log_interval = 1000): 
    ''' Extracts descriptions & relations for all movies.'''
    
    print('Extracting character descriptions & relations...')

    # Get all xml files in the directory
    xml_files = [f for f in os.listdir(CORENLP_OUTPUT_DIR) if f.endswith('.xml')]
    num_files = len(xml_files)
    
    # Create dataframes to store the extracted descriptions and relations
    descriptions = pd.DataFrame(columns=['movie_id', 'character', 'agent_verbs', 'patient_verbs', 'attributes', 'title', 'religion', 'age'])
    relations = pd.DataFrame(columns=['movie_id', 'subject', 'object'])

    for idx, xml_file in enumerate(xml_files):
        # Print progress every log_interval files
        if idx % log_interval == 0:
            print('Progress: {}/{} ({}%)'.format(idx, num_files, round(idx/num_files*100, 2)))

        # Parse the xml file and extract description & relations dataframes
        tree = ET.parse(os.path.join(CORENLP_OUTPUT_DIR, xml_file))
        descriptions_df, relations_df = get_descriptions_relations(tree)

        # Concatenate descriptions to previous descriptions
        descriptions = pd.concat([descriptions, descriptions_df], ignore_index=True)
        relations = pd.concat([relations, relations_df], ignore_index=True)

    # Save descriptions to a csv file
    descriptions.to_csv('Data/CoreNLP/descriptions.csv', sep='\t')
    relations.to_csv('Data/CoreNLP/relations.csv', sep='\t')

    return descriptions, relations
