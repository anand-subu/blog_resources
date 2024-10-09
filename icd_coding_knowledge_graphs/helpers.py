import simple_icd_10_cm as cm
from openai import OpenAI
from tqdm import tqdm
import random
import json
import scispacy
from scispacy.candidate_generation import CandidateGenerator
from scispacy.linking import EntityLinker
import networkx as nx

candidate_generator = CandidateGenerator(name="umls")
entity_linker = EntityLinker(resolve_abbreviations=True, name="umls", candidate_generator=candidate_generator)

client = OpenAI(
    api_key="",
)

prompt_relation_extraction = """You are an expert medical professional, qualified in ICD coding and medical terminology.
You are given a description of an ICD code, and your task is to extract relevant entities and construct a graph by identifying the relationships between these entities. 

Follow these detailed instructions:

Step 1: Extract Entities
Identify and extract the following entity types from the provided ICD code description.

Instructions: 
1. Do not output any placeholders for entity types that are not mentioned in the description.
2. You must separate entities of the same type with ||.
3. Strictly do not output content that is not present in the description.

Entity Types:
1. condition: Identify the medical conditions and/or injuries described by the ICD code.
2. bodypart: Identify any specific body parts mentioned affected by the conditions.
3. severity: Determine the severity or degree or stage of the conditions if mentioned (e.g., first degree, second degree, third degree, mild, moderate, severe, type I, type II).
4. encounter_type: Identify the type of medical encounter described (e.g., initial encounter, subsequent encounter, sequela).
5. cause: Extract the cause or reasons for the conditions or injuries (e.g., foreign object, poisoning, burn, fall, collision).
6. laterality: Identify if a specific side of the body is affected (e.g., left, right, unspecified).
7. person: Represents the person affected (e.g., unspecified person, suspect, bystander).
8. fetus: Represents the fetus affected in maternal care cases.
9. procedure: Represents any treatment or procedures associated with the conditions (e.g., surgery, prosthesis).
10. complication: Represents any complications arising from the conditions or treatments (e.g., nonunion, delayed healing).
11. trimester: Represents the trimester of pregnancy if applicable.
12. other_info: Represents any other important medical information that are not covered by the above entities.


Step 2: Construct Relationships

From the extracted entities, identify and extract the entities that are related to each other.
Some examples of relations are:
- Example: "Dislocation" (condition) affects "knee" (bodypart).
- Example: "Burn" (condition) has "third degree" (severity).
- Example: "Pre-eclampsia" (condition) occurs during "second trimester" (trimester).
- Example: "Laceration" (condition) caused by "sharp object" (cause).
- Example: "hand" (bodypart) affected is "right" (laterality).
- Example: "Injury" (condition) affects "unspecified person" (person).
- Example: "Chorioamnionitis" (condition) affects "fetus" (fetus).
- Example: "Gastric band complication" (condition) treated by "gastric band procedure" (procedure).
- Example: "Dislocation" (condition) occurs during "initial encounter" (encounter_type).
- Example: "Burn" (condition) has "delayed healing" (complication).
   
Represent each pair of related entities in the format (Entity 1||Entity 2).
Some examples of input and output are provided below:

### Input
ICD Code Description: Laceration with foreign body of right breast, initial encounter

### Output
Entities:
condition: Laceration
bodypart: breast
encounter_type: initial encounter
cause: foreign body
laterality: right

Relations:
(Laceration||foreign body)
(Laceration||breast)
(right||breast)
(Laceration||initial encounter)

### Input
ICD Code Description: Nondisplaced segmental fracture of shaft of radius, unspecified arm, subsequent encounter for open fracture type I or II with delayed healing

### Output
Entities:
condition: Nondisplaced segmental fracture||open fracture
bodypart: shaft of radius||arm
encounter_type: subsequent encounter
severity: type I||type II
complication: delayed healing
laterality: unspecified

Relations:
(Nondisplaced segmental fracture||shaft of radius)
(Nondisplaced segmental fracture||arm)
(unspecified||arm)
(Nondisplaced segmental fracture||subsequent encounter)
(subsequent encounter||open fracture)
(open fracture||type I)
(open fracture||type II)
(open fracture||delayed healing)

### Input
ICD Code Description: Displaced oblique fracture of shaft of left fibula, subsequent encounter for open fracture type IIIA, IIIB, or IIIC with malunion

### Output
Entities:
condition: Displaced oblique fracture||open fracture
bodypart: shaft||fibula
encounter_type: subsequent encounter
severity: type IIIA||type IIIB||type IIIC
complication: malunion
laterality: left

Relations:
(Displaced oblique fracture||shaft)
(shaft||fibula)
(left||fibula)
(Displaced oblique fracture||subsequent encounter)
(subsequent encounter||open fracture)
(open fracture||type IIIA)
(open fracture||type IIIB)
(open fracture||type IIIC)
(open fracture||malunion)

### Input
ICD Code Description: Maternal care for hydrops fetalis, second trimester, fetus 2

### Output
Entities:
condition: hydrops fetalis
trimester: second trimester
fetus: fetus 2

Relations:
(hydrops fetalis||second trimester)
(hydrops fetalis||fetus 2)"""

def get_completion(prompt, input, model="gpt-4o-mini", temperature=0.0):
    """
    Generate a completion for a given prompt and input using a specified model.

    Args:
    prompt (str): The initial context or setup for the AI to generate a response.
    input (str): The user input that follows the prompt.
    model (str, optional): The model identifier to be used for generating the completion. Default is 'gpt-4o-mini'.
    temperature (float, optional): The creativity or randomness level of the response. Default is 0.0.

    Returns:
    str: The content of the message generated by the model as a response to the user input.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": input}],
        temperature=temperature,
    )
    return response.choices[0].message.content

def get_max_similarity_concept_id(candidates):
    """
    Determines the concept ID with the maximum similarity score from a list of candidate concepts.

    Args:
    candidates (list): A list of MentionCandidate objects, each containing a 'similarities' list and a 'concept_id'.

    Returns:
    any: The concept ID associated with the highest similarity score in the given candidates list, or None if the list is empty.
    """
    max_similarity = float('-inf')
    best_concept_id = None

    for candidate in candidates:
        candidate_max_similarity = max(candidate.similarities)
        
        if candidate_max_similarity > max_similarity:
            max_similarity = candidate_max_similarity
            best_concept_id = candidate.concept_id

    return best_concept_id

def normalize_entities(entities):
    """
    Extracts the top linked concept for the first recognized entity in a given text using scispaCy.

    Parameters:
    - text (str): The input text to extract entities from.

    Returns:
    - str: The normalized entity if identified, or the original entity text.
    """
    candidates = candidate_generator(entities, k = 5)
    normalized_entity_map = {}
    
    for entity, candidate in tqdm(zip(entities, candidates)):
        most_likely_cui = get_max_similarity_concept_id(candidate)
        cui_entity = entity_linker.kb.cui_to_entity[most_likely_cui]
        normalized_entity_map[entity] = cui_entity.canonical_name
    
    return normalized_entity_map

def extract_entities(input_text):
    """
    Extracts entity names from a formatted text input. The input text should contain sections
    labeled 'Entities:' and 'Relations:' which denote the start of entity and relation lists, respectively.

    Args:
    input_text (str): A string containing the text input from which entities will be extracted. The
                      text should be formatted with sections named 'Entities:' followed by entity entries,
                      and 'Relations:', although only 'Entities:' section is utilized in this function.

    Returns:
    list: A list of strings, each representing an entity name extracted from the 'Entities:' section of the input.
    """    
    entities_list = []
    relations = []
    lines = input_text.strip().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if line == 'Entities:':
            current_section = 'entities'
            continue
        elif line == 'Relations:':
            current_section = 'relations'
            continue
        elif line == '':
            continue

        if current_section == 'entities':
            if ':' in line:
                entity_type, entity_name = line.split(':', 1)
                entity_type = entity_type.strip()
                entity_names = entity_name.split("||")
                entity_names = [x.strip() for x in entity_names]
                for entity_name in entity_names:
                    entities_list.append(entity_name)
    return entities_list


def parse_entities(lines, normalized_entity_map):
    """
    Parses entity information from the provided lines.

    Args:
        lines (list): List of strings containing entity data in the format "EntityType: EntityName1 || EntityName2".
        normalized_entity_map (dict): Dictionary mapping original entity names to their normalized forms.

    Returns:
        tuple: A tuple containing:
            - entities (dict): A dictionary where keys are normalized entity names and values are entity types.
            - overall_entities (set): A set of all entity names encountered.
    """    
    entities = {}
    overall_entities = set()
    for line in lines:
        if ':' in line:
            entity_type, entity_names = line.split(':', 1)
            entity_type = entity_type.strip()
            entity_names = [name.strip() for name in entity_names.split("||")]
            for entity_name in entity_names:
                overall_entities.add(entity_name)
                entities[normalized_entity_map[entity_name]] = entity_type
    return entities, overall_entities

def parse_relations(lines):
    """
    Parses relation information from the provided lines.

    Args:
        lines (list): List of strings containing relation data in the format "(EntityName1 || EntityName2)".

    Returns:
        list: A list of tuples representing relations, where each tuple is a pair of related entity names.
    """    
    relations = []
    for line in lines:
        if line.startswith('(') and line.endswith(')'):
            entity_names = [e.strip() for e in line[1:-1].split('||')]
            if len(entity_names) == 2 and all(entity_names):
                relations.append((entity_names[0], entity_names[1]))
    return relations

def build_graph(input_text, icd_code, icd_description, normalized_entity_map):
    """
    Builds a graph based on entity and relation information extracted from input text.

    Args:
        input_text (str): Text containing entity and relation information.
        icd_code (str): ICD code for the graph root node.
        icd_description (str): Description of the ICD code.
        normalized_entity_map (dict): Dictionary mapping original entity names to their normalized forms.

    Returns:
        networkx.Graph: A NetworkX graph representing the entities, relations, and the ICD root node.
    """    
    lines = input_text.strip().split('\n')
    entities_section, relations_section = split_sections(lines)
    
    entities, overall_entities = parse_entities(entities_section, normalized_entity_map)
    relations = parse_relations(relations_section)
    
    G = create_graph(icd_code, icd_description, entities, overall_entities, relations, normalized_entity_map)
    return G

def split_sections(lines):
    """
    Splits the input lines into separate sections for entities and relations.

    Args:
        lines (list): List of strings containing the lines of the input text.

    Returns:
        tuple: A tuple containing:
            - entities_section (list): List of strings corresponding to the "Entities:" section.
            - relations_section (list): List of strings corresponding to the "Relations:" section.
    """
    entities_section = []
    relations_section = []
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line == 'Entities:':
            current_section = entities_section
        elif line == 'Relations:':
            current_section = relations_section
        elif line and current_section is not None:
            current_section.append(line)
    
    return entities_section, relations_section

def create_graph(icd_code, description, entities, overall_entities, relations, normalized_entity_map):
    """
    Creates a NetworkX graph using the provided ICD code, entities, and relations.

    Args:
        icd_code (str): ICD code for the root node of the graph.
        description (str): Description of the ICD code.
        entities (dict): Dictionary of entities with their types.
        overall_entities (set): Set of all entity names encountered.
        relations (list): List of tuples representing relations between entities.
        normalized_entity_map (dict): Dictionary mapping original entity names to their normalized forms.

    Returns:
        networkx.Graph: A NetworkX graph representing the entities, relations, and the ICD root node.
    """    
    G = nx.Graph()
    G.add_node(icd_code, type="ICD", description = description)
    
    for entity_name, entity_type in entities.items():
        G.add_node(entity_name, type=entity_type)
        G.add_edge(entity_name, icd_code)
    
    for entity1, entity2 in relations:
        if entity1 in overall_entities and entity2 in overall_entities:
            entity1_normalized = normalized_entity_map[entity1]
            entity2_normalized = normalized_entity_map[entity2]
            G.add_edge(entity1_normalized, entity2_normalized)
    
    return G
