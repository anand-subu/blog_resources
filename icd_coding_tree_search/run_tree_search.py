import argparse
import os
import json
from tree_search_icd import get_icd_codes
from tqdm import tqdm

def process_medical_notes(input_dir, output_file, model_name):
    code_map = {}
    # Ensure the input directory is valid
    if not os.path.isdir(input_dir):
        raise ValueError("The specified input directory does not exist.")

    # Process each file in the input directory
    for files in tqdm(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, files)
        with open(file_path, "r", encoding="utf-8") as file:
            medical_note = file.read()
        
        icd_codes = get_icd_codes(medical_note, model_name)
        code_map[files] = icd_codes

    # Save the ICD codes to a JSON file
    with open(output_file, "w") as f:
        json.dump(code_map, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process medical notes to extract ICD codes using a specified model.")
    parser.add_argument("--input_dir", help="Directory containing the medical text files")
    parser.add_argument("--output_file", help="File to save the extracted ICD codes in JSON format")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613", help="Model name to use for ICD code extraction")

    args = parser.parse_args()
    process_medical_notes(args.input_dir, args.output_file, args.model_name)
