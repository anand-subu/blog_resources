import os
import argparse

def clean_markdown_code_blocks(content):
    """
    Removes markdown code block identifiers from the given content.
    
    Parameters:
        content (list[str]): Lines of a markdown file.
    
    Returns:
        list[str]: Cleaned lines without ```markdown and ``` blocks.
    """
    return [line for line in content if not line.strip().startswith("```")]

def process_markdown_files(input_dir, output_dir):
    """
    Processes markdown files in the input directory by removing code block markers
    and saves cleaned versions to a mirrored structure in the output directory.

    Parameters:
        input_dir (str): Path to the source directory containing markdown files.
        output_dir (str): Path to the destination directory for cleaned files.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".md"):
                input_path = os.path.join(root, file)

                # Construct the mirrored output path
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Read, clean, and write the file
                with open(input_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                cleaned_lines = clean_markdown_code_blocks(lines)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.writelines(cleaned_lines)

    print(f"Processed markdown files saved to '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Clean markdown files by removing code block identifiers.")
    parser.add_argument("input_dir", help="Path to the input directory containing markdown files.")
    parser.add_argument("output_dir", help="Path to the output directory for cleaned markdown files.")
    
    args = parser.parse_args()

    process_markdown_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
