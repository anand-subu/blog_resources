import argparse
import os
from helpers import build_translation_prompt, get_response
from tqdm import tqdm

def translate_directory(input_dir, output_dir, model_name):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    translated_files = {}
    # Process each file in the input directory
    for item in tqdm(os.listdir(input_dir)):
        input_path = os.path.join(input_dir, item)
        with open(input_path, "r", encoding="utf-8") as file:
            input_note = file.read()
        
        translation_prompt = build_translation_prompt(input_note)
        translated_note = get_response(translation_prompt, model_name=model_name)
        translated_files[item] = translated_note
        
    # Save the translated files to the output directory
    for key, value in translated_files.items():
        output_path = os.path.join(output_dir, key)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text files from one directory to another using a specified model.")
    parser.add_argument("--input_dir", help="Directory containing text files to be translated")
    parser.add_argument("--output_dir", help="Directory to save the translated text files")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", help="Model name to use for translation")

    args = parser.parse_args()
    translate_directory(args.input_dir, args.output_dir, args.model_name)
