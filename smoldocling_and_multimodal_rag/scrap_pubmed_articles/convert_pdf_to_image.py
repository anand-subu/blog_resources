import os
import argparse
from tqdm import tqdm
from pdf2image import convert_from_path

def pdf_to_png(pdf_path, output_folder_base, poppler_path=None):
    """
    Convert a single PDF file into PNG images.

    Args:
        pdf_path (str): The path to the PDF file to be converted.
        output_folder_base (str): The base output folder where the PNG images will be stored.
        poppler_path (str, optional): Path to the Poppler binary files, if not on system PATH.
    """
    # Create a subfolder named after the PDF file (without extension)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(output_folder_base, pdf_name)
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to a list of images
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    # Save each page as a separate PNG
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')
        print(f"Saved: {image_path}")

def process_pdf_folder(input_folder, output_folder_base, poppler_path=None):
    """
    Process all PDF files in the specified input folder.

    Args:
        input_folder (str): Path to the folder containing PDF files.
        output_folder_base (str): Base folder to which converted PNG images will be saved.
        poppler_path (str, optional): Path to the Poppler binaries, if needed.
    """
    # Iterate through all PDF files in the input folder
    for filename in tqdm(os.listdir(input_folder), desc="Converting PDFs", unit="file"):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {pdf_path}")
            pdf_to_png(pdf_path, output_folder_base, poppler_path=poppler_path)

def main():
    """
    Main function to parse command-line arguments and execute the PDF-to-PNG conversion process.
    """
    parser = argparse.ArgumentParser(
        description="Convert PDFs in a folder to PNG images."
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Folder containing PDF files."
    )
    parser.add_argument(
        "--output_folder_base",
        type=str,
        required=True,
        help="Base folder to store output PNG images."
    )
    parser.add_argument(
        "--poppler_path",
        type=str,
        default=None,
        help="Path to Poppler binaries (if not on system PATH)."
    )

    args = parser.parse_args()

    process_pdf_folder(
        input_folder=args.input_folder,
        output_folder_base=args.output_folder_base,
        poppler_path=args.poppler_path
    )

if __name__ == "__main__":
    main()
