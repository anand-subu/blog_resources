import json
from tqdm import tqdm

SYSTEM_PROMPT = """The Medical Subject Headings (MeSH) thesaurus is a controlled and hierarchically-organized vocabulary produced by the National Library of Medicine. It is used for indexing, cataloging, and searching of biomedical and health-related information. 
You are provided with text from  a biomedical document. Extract all chemical and disease entities, and predict the MeSH identifiers for each entity.
Do not predict entities that are not present in the provided input text.
If an entity has multiple MeSH identifiers, provide all identifiers separated by a "|".
If an entity has no MeSH identifier, respond with None."""
# definition for first part of prompt borrowed from https://www.nlm.nih.gov/mesh/meshhome.html

SYSTEM_RAG_PROMPT = """The Medical Subject Headings (MeSH) thesaurus is a controlled and hierarchically-organized vocabulary produced by the National Library of Medicine. It is used for indexing, cataloging, and searching of biomedical and health-related information. 
You are provided with text from  a biomedical document. MeSH identifiers that might be relevant to the input along with information about their definitions, relevant entity names and aliases are provided as context.
Extract all chemical and disease entities, along with the MeSH identifiers for each entity. Strictly perform this task based on the provided context alone. 
Do not predict MeSH IDs that are not present in the provided context.
Do not predict entities that are not present in the provided input text.
If an entity has multiple MeSH identifiers, provide all identifiers separated by a "|".
If an entity has no MeSH identifier, respond with None.
An example of how to format your output is provided:
Entity:<Entity Name1>;MeSH ID:<MeSH IDs>
Entity:<Entity Name2>;MeSH ID:<MeSH IDs>
.
.
Entity:<Entity NameN>;MeSH ID:<MeSH IDs>"""

def build_few_shot_prompt(system_prompt, content, few_shot_examples):
    """
    Builds the few-shot prompt using provided examples.

    Args:
        system_prompt (str): Task description for the LLM
        content (dict): The content for which to create a query, similar to the
            structure required by `create_query`.
        few_shot_examples (list of dict): Examples to simulate a hypothetical
            conversation. Each dict must have "options" and an "answer".

    Returns:
        list of dict: A list of messages, simulating a conversation with
            few-shot examples, followed by the current user query.
    """
    messages = []
    for idx,item in enumerate(few_shot_examples):
        if idx == 0:
            messages.append({"role": "user", "content": system_prompt + "\n" + create_input(item)})
        else:
            messages.append({"role": "user", "content": create_input(item)})
            
        messages.append({"role": "assistant", "content": create_output(item)})
    messages.append({"role": "user", "content": create_input(content)})
    
    return messages

def build_entity_prompt(content):
    """
    Builds a prompt for entity extraction from a biomedical text.

    Args:
    - content (dict): A dictionary with keys 'title' and 'abstract'.

    Returns:
    - list of dict: A list containing a single dictionary with keys 'role' and 'content',
      where 'content' combines a system prompt, an entity extraction prompt, and the biomedical text.
    """
    system_prompt = "Answer the question factually and precisely."
    entity_prompt = "What are the chemical and disease related entities present in this biomedical text?"
    return [{"role": "user", "content": system_prompt + "\n" + entity_prompt + "\n" + content["title"] + " " + content["abstract"]}]

def build_rag_prompt(system_prompt, content, context):
    """
    Builds a prompt for the retrieval-augmented generation (RAG) process.

    Args:
    - system_prompt (str): The initial prompt for the RAG system.
    - content (dict): A dictionary with the content to be processed, expected to have 'title' and 'abstract'.
    - context (list of dict): A list of context items to be formatted and included in the prompt.

    Returns:
    - list of dict: A list containing a single dictionary with 'role' and 'content',
      where 'content' includes the system prompt, relevant context, and input content.
    """
    return [{"role": "user", "content": system_prompt + "\nRelevant Context:\n" + "\n".join([format_context(x) for x in context]) + "\nInput:\n" + create_input(content)}]

def parse_entities_from_trained_model(content):
    """
    Extracts a list of entities from the output of a trained model.

    Args:
    - content (str): The raw string output from a trained model.

    Returns:
    - list of str: A list of entities extracted from the model's output.
    """
    return content.split("The entities are:")[-1].split(",")

def read_jsonl_file(file_path):
    """
    Parses a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list of dict: A list where each element is a dictionary representing
            a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

def parse_dataset(file_path):
    """
    Parse the BioCreative Dataset.

    Args:
    - file_path (str): Path to the file containing the documents.

    Returns:
    - list of dict: A list where each element is a dictionary representing a document.
    """
    documents = []
    current_doc = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if "|t|" in line:
                if current_doc:
                    documents.append(current_doc)
                id_, title = line.split("|t|", 1)
                current_doc = {'id': id_, 'title': title, 'abstract': '', 'annotations': []}
            elif "|a|" in line:
                _, abstract = line.split("|a|", 1)
                current_doc['abstract'] = abstract
            else:
                parts = line.split("\t")
                if parts[1] == "CID":
                    continue
                annotation = {
                    'text': parts[3],
                    'type': parts[4],
                    'identifier': parts[5]
                }
                current_doc['annotations'].append(annotation)

        if current_doc:
            documents.append(current_doc)

    return documents

def deduplicate_annotations(documents):
    """
    Filter documents to ensure annotation consistency.

    Args:
    - documents (list of dict): The list of documents to be checked.
    """
    for doc in documents:
        doc["annotations"] = remove_duplicates(doc["annotations"])
        
def remove_duplicates(dict_list):
    """
    Remove duplicate dictionaries from a list of dictionaries.

    Args:
    - dict_list (list of dict): A list of dictionaries from which duplicates are to be removed.

    Returns:
    - list of dict: A list of dictionaries after removing duplicates.
    """
    unique_dicts = []  
    seen = set()

    for d in dict_list:
        dict_tuple = tuple(sorted(d.items()))
        if dict_tuple not in seen:
            seen.add(dict_tuple)
            unique_dicts.append(d)

    return unique_dicts

def create_input(item):
    """
    Concatenates the title and abstract of an item into a single string.

    Args:
    - item (dict): A dictionary with keys 'title' and 'abstract'.

    Returns:
    - str: A single string combining the title and abstract separated by a newline.
    """
    return item["title"] + " " + item["abstract"]

def create_output(item):
    """
    Generates a formatted string representation of annotations within an item.

    This function processes each annotation in the item's 'annotations' list, extracting the
    text and the MeSH ID (Medical Subject Headings Identifier). If the MeSH ID is "-1", it is replaced with "None".

    Args:
    - item (dict): A dictionary containing a list of annotations under the key 'annotations'.
                   Each annotation is a dictionary with keys 'text' and 'identifier'.

    Returns:
    - str: A newline-separated string where each line represents an annotation in the format "Entity:<text>;MeSH ID:<MeSH ID>".
    """
    output = []
    
    for annotation in item["annotations"]:
        if annotation["identifier"] == "-1":
            mesh_id = "None"
        else:
            mesh_id = annotation["identifier"]
        
        output.append("Entity:" + annotation["text"] + ";" + "MeSH ID:" + mesh_id)
    
    return "\n".join(output)



def read_jsonl_file(file_path):
    """
    Parses a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list of dict: A list where each element is a dictionary representing
            a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

def write_jsonl_file(file_path, dict_list):
    """
    Write a list of dictionaries to a JSON Lines file.

    Args:
    - file_path (str): The path to the file where the data will be written.
    - dict_list (list): A list of dictionaries to write to the file.
    """
    with open(file_path, 'w') as file:
        for dictionary in dict_list:
            # Convert the dictionary to a JSON string and write it to the file.
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def parse_answer(model_output):
    """
    Parses a string output from a model that contains entities and their corresponding MeSH IDs,
    and returns a dictionary mapping entities to MeSH IDs.

    The function expects each line in the model output to be formatted as "Entity:<entity>;MeSH ID:<MeSH ID>".
    Lines that do not conform to this format are ignored. It is also assumed that each entity appears only once
    in the output. If an entity appears multiple times with different MeSH IDs, only the last one will be retained
    in the returned dictionary.

    Args:
    - model_output (str): The string output from the model, containing entities and their MeSH IDs.

    Returns:
    - dict: A dictionary where keys are entities (str) and values are the corresponding MeSH IDs (str).
            Entities are trimmed of leading and trailing whitespace, ensuring clean data.

    Note:
    - The function performs basic error checking and will skip lines that do not strictly conform to the expected format.
    - In cases of duplicate entities, the function currently overwrites previous entries with the latest MeSH ID.
      Consider modifying the function if a different handling of duplicates is required (e.g., aggregating MeSH IDs into a list).
    """
    entity_mesh_mapping = []
    for item in model_output.split("\n"):
        entity_prediction = {}
        if "Entity" not in item or "MeSH ID" not in item:
            continue

        split_info = item.split(";")
        if len(split_info) != 2:
            continue

        entity_info, mesh_info = split_info
        entity_split = entity_info.split(":")
        mesh_split = mesh_info.split(":")
        
        if len(entity_split) != 2 or len(mesh_split) != 2:
            continue
        
        _, entity_pred = entity_split
        _, mesh_ids = mesh_split

        # Strip whitespace to ensure clean data
        entity_pred = entity_pred.strip()
        mesh_ids = mesh_ids.strip()

        # If handling duplicates, consider appending to a list instead
        entity_prediction["text"] = entity_pred
        entity_prediction["identifier"] = mesh_ids
        entity_mesh_mapping.append(entity_prediction)

    return entity_mesh_mapping

def format_context(data):
    """
    Formats a dictionary containing medical concept information into a multi-line string.

    Args:
    - data (dict): A dictionary with the following keys:
        - concept_id: A string representing the MeSH (Medical Subject Headings) identifier.
        - canonical_name: A string representing the canonical name of the medical concept.
        - aliases: A list of strings representing alternative names or identifiers for the concept.
        - definition: A string providing a brief description or definition of the concept.

    Returns:
    - str: A formatted multi-line string that presents the concept information in a readable format.

    The returned string includes the MeSH ID, canonical name, a list of aliases, and the definition of the concept,
    each on a new line.
    """
    placeholder_string = """MeSH ID: {concept_id}
Definition: {definition}
Canonical Name: {canonical_name}
Aliases: {aliases}
"""
    return placeholder_string.format_map(data)

def calculate_entity_metrics(gt, pred):
    """
    Calculate precision, recall, and F1-score for entity recognition.

    Args:
    - gt (list of dict): Ground truth data
    - pred (list of dict): Predicted data

    Returns:
    tuple: A tuple containing precision, recall, and F1-score (in that order).
    """
    ground_truth_set = set([x["text"].lower() for x in gt])
    predicted_set = set([x["text"].lower() for x in pred])

    # True positives are predicted items that are in the ground truth
    true_positives = len(predicted_set.intersection(ground_truth_set))
    
    # Precision calculation
    if len(predicted_set) == 0:
        precision = 0
    else:
        precision = true_positives / len(predicted_set)
    
    # Recall calculation
    if len(ground_truth_set) == 0:
        recall = 0
    else:
        recall = true_positives / len(ground_truth_set)
    
    # F1-score calculation
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def calculate_mesh_metrics(gt, pred):
    """
    Calculate precision, recall, and F1-score for matching MeSH (Medical Subject Headings) codes.

    Args:
    - gt (list of dict): Ground truth data
    - pred (list of dict): Predicted data

    Returns:
    tuple: A tuple containing precision, recall, and F1-score (in that order).
    """
    ground_truth = []

    for item in gt:
        mesh_codes = item["identifier"]
        if mesh_codes == "-1":
            mesh_codes = "None"
        mesh_codes_split = mesh_codes.split("|")
        for elem in mesh_codes_split:
            combined_elem = {"entity": item["text"].lower(), "identifier": elem}
            if combined_elem not in ground_truth:
                ground_truth.append(combined_elem)
    
    predicted = []
    for item in pred:
        mesh_codes = item["identifier"]
        mesh_codes_split = mesh_codes.strip().split("|")
        for elem in mesh_codes_split:
            combined_elem = {"entity": item["text"].lower(), "identifier": elem}
            if combined_elem not in predicted:
                predicted.append(combined_elem)
    # True positives are predicted items that are in the ground truth
    true_positives = len([x for x in predicted if x in ground_truth])
    
    # Precision calculation
    if len(predicted) == 0:
        precision = 0
    else:
        precision = true_positives / len(predicted)
    
    # Recall calculation
    if len(ground_truth) == 0:
        recall = 0
    else:
        recall = true_positives / len(ground_truth)
    
    # F1-score calculation
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def process_index(index):
    """
    Processes the initial document index to combine aliases, canonical names, and definitions into a single text index.

    Args:
    - index (Dict): The MeSH knowledge base
    Returns:
        List[List[int, str]]: A dictionary with document IDs as keys and combined text indices as values.
    """
    processed_index = []
    for key, value in tqdm(index.items()):
        assert(type(value["aliases"]) != list)
        aliases_text = " ".join(value["aliases"].split(","))
        text_index = (aliases_text + " " +  value.get("canonical_name", "")).strip()
        if "definition" in value:
            text_index += " " + value["definition"]
        processed_index.append([value["concept_id"], text_index])
    return processed_index

def process_mesh_kb(kb):
    """
    Helper function for processing alias types in the MeSH KB. Some of the aliases in the KB are present as a list and some others are comma-separated
    
    Args:
    - kb (list of dict): The MeSH KB elements
    """
    
    for item in kb:
        if isinstance(item["aliases"], list):
            item["aliases"] = ",".join(item["aliases"])
