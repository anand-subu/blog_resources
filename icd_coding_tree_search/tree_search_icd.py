from helpers import *

def get_icd_codes(medical_note, model_name="gpt-3.5-turbo-0613", temperature=0.0):
    """
    Identifies relevant ICD-10 codes for a given medical note by querying a language model.

    This function implements the tree-search algorithm for ICD coding described in https://openreview.net/forum?id=mqnR8rGWkn.

    Args:
        medical_note (str): The medical note for which ICD-10 codes are to be identified.
        model_name (str): The identifier for the language model used in the API (default is 'gpt-3.5-turbo-0613').

    Returns:
        list of str: A list of confirmed ICD-10 codes that are relevant to the medical note.
    """
    assigned_codes = []
    candidate_codes = [x.name for x in CHAPTER_LIST]
    parent_codes = []
    prompt_count = 0

    while prompt_count < 50:
        code_descriptions = {}
        for x in candidate_codes:
            description, code = get_name_and_description(x, model_name)
            code_descriptions[description] = code

        prompt = build_zero_shot_prompt(medical_note, list(code_descriptions.keys()), model_name=model_name)
        lm_response = get_response(prompt, model_name, temperature=temperature, max_tokens=500)
        predicted_codes = parse_outputs(lm_response, code_descriptions, model_name=model_name)

        for code in predicted_codes:
            if cm.is_leaf(code["code"]):
                assigned_codes.append(code["code"])
            else:
                parent_codes.append(code)

        if len(parent_codes) > 0:
            parent_code = parent_codes.pop(0)
            candidate_codes = cm.get_children(parent_code["code"])
        else:
            break

        prompt_count += 1

    return assigned_codes
