import re
import simple_icd_10_cm as cm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from prompt_templates import *

CHAPTER_LIST = cm.chapter_list

client = OpenAI()

def construct_translation_prompt(medical_note):
    """
    Construct a prompt template for translating spanish medical notes to english.
    
    Args:
        medical_note (str): The medical case note.
        
    Returns:
        str: A structured template ready to be used as input for a language model.
    """    
    translation_prompt = """You are an expert Spanish-to-English translator. You are provided with a clinical note written in Spanish.
You must translate the note into English. You must ensure that you properly translate the medical and technical terms from Spanish to English without any mistakes.
Spanish Medical Note:
{medical_note}"""
    
    return translation_prompt.format(medical_note = medical_note)

def build_translation_prompt(input_note, system_prompt=""):
    """
    Build a zero-shot prompt for translating spanish medical notes to english.
    
    Args:
        input_note (str): The input note or query.
        system_prompt (str): Optional initial system prompt or instruction.
        
    Returns:
        list of dict: A structured list of dictionaries defining the role and content of each message.
    """
    input_prompt = construct_translation_prompt(input_note)
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]

def remove_extra_spaces(text):
    """
    Remove extra spaces from a given text.
    
    Args:
        text (str): The original text string.
        
    Returns:
        str: The cleaned text with extra spaces removed.
    """
    return re.sub(r'\s+', ' ', text).strip()

def remove_last_parenthesis(text):
    """
    Removes the last occurrence of content within parentheses from the provided text.

    Args:
    text (str): The input string from which to remove the last parentheses and its content.

    Returns:
    str: The modified string with the last parentheses content removed.
    """
    pattern = r'\([^()]*\)(?!.*\([^()]*\))'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def format_code_descriptions(text, model_name):
    """
    Format the ICD-10 code descriptions by removing content inside brackets and extra spaces.
    
    Args:
        text (str): The original text containing ICD-10 code descriptions.
        
    Returns:
        str: The cleaned text with content in brackets removed and extra spaces cleaned up.
    """
    pattern = r'\([^()]*\)(?!.*\([^()]*\))'
    cleaned_text = remove_last_parenthesis(text)
    cleaned_text = remove_extra_spaces(cleaned_text)
        
    return cleaned_text

def construct_prompt_template(case_note, code_descriptions, model_name):
    """
    Construct a prompt template for evaluating ICD-10 code descriptions against a given case note.
    
    Args:
        case_note (str): The medical case note.
        code_descriptions (str): The ICD-10 code descriptions formatted as a single string.
        
    Returns:
        str: A structured template ready to be used as input for a language model.
    """
    template = prompt_template_dict[model_name]

    return template.format(note=case_note, code_descriptions=code_descriptions)

def build_zero_shot_prompt(input_note, descriptions, model_name, system_prompt=""):
    """
    Build a zero-shot classification prompt with system and user roles for a language model.
    
    Args:
        input_note (str): The input note or query.
        descriptions (list of str): List of ICD-10 code descriptions.
        system_prompt (str): Optional initial system prompt or instruction.
        
    Returns:
        list of dict: A structured list of dictionaries defining the role and content of each message.
    """
    if model_name == "meta-llama/Llama-2-70b-chat-hf":
        code_descriptions = "\n".join(["* " + x for x in descriptions])
    else:
        code_descriptions = "\n".join(descriptions)

    input_prompt = construct_prompt_template(input_note, code_descriptions, model_name)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}]

def get_response(messages, model_name, temperature=0.0, max_tokens=500):
    """
    Obtain responses from a specified model via the chat-completions API.
    
    Args:
        messages (list of dict): List of messages structured for API input.
        model_name (str): Identifier for the model to query.
        temperature (float): Controls randomness of response, where 0 is deterministic.
        max_tokens (int): Limit on the number of tokens in the response.
        
    Returns:
        str: The content of the response message from the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def remove_noisy_prefix(text):
    # Removing numbers or letters followed by a dot and optional space at the beginning of the string
    cleaned_text = text.replace("* ", "").strip()
    cleaned_text = re.sub(r"^\s*\w+\.\s*", "", cleaned_text)
    return cleaned_text.strip()

def parse_outputs(output, code_description_map, model_name):
    """
    Parse model outputs to confirm ICD-10 codes based on a given description map.
    
    Args:
        output (str): The model output containing confirmations.
        code_description_map (dict): Mapping of descriptions to ICD-10 codes.
        
    Returns:
        list of dict: A list of confirmed codes and their descriptions.
    """
    confirmed_codes = []
    split_outputs = [x for x in output.split("\n") if x]
    for item in split_outputs:
        try:                
            code_description, confirmation = item.split(":", 1)
            if model_name == "meta-llama/Llama-2-70b-chat-hf":
                code_description = remove_noisy_prefix(code_description)

            if confirmation.lower().strip().startswith("yes"):
                try:
                    code = code_description_map[code_description]
                    confirmed_codes.append({"code": code, "description": code_description})
                except Exception as e:
                    print(str(e) + " Here")
                    continue
        except:
            continue
    return confirmed_codes

def get_name_and_description(code, model_name):
    """
    Retrieve the name and description of an ICD-10 code.
    
    Args:
        code (str): The ICD-10 code.
        
    Returns:
        tuple: A tuple containing the formatted description and the name of the code.
    """
    full_data = cm.get_full_data(code).split("\n")
    return format_code_descriptions(full_data[3], model_name), full_data[1]
