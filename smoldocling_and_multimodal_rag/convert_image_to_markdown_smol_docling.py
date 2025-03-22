import os
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument


def get_device():
    """
    Returns the available device: 'cuda' if available, else 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_processor(device):
    """
    Loads the HuggingFace model and processor for SmolDocling.

    Args:
        device (str): 'cuda' or 'cpu'

    Returns:
        tuple: (processor, model)
    """
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16,
    ).to(device)
    return processor, model


def process_image_to_docling_markdown(image, processor, model, device):
    """
    Converts an image of a document page into Docling-style Markdown.

    Args:
        image (PIL.Image or np.array): Input image of the document page.
        processor: HuggingFace processor for the vision-to-seq model.
        model: Pretrained HuggingFace model for vision-to-seq tasks.
        device (str): 'cuda' or 'cpu', determines where inference runs.

    Returns:
        str: Markdown string representing the Docling conversion of the image.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."}
            ]
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=8192)
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]

    doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])

    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)

    return doc.export_to_markdown()


def process_image_folders(root_folder, output_root, processor, model, device):
    """
    Walks through nested folders of images, converts each to Docling Markdown,
    and saves results in a mirrored folder structure.

    Args:
        root_folder (str or Path): Path to the root folder containing image folders.
        output_root (str or Path): Root directory to save converted Markdown files.
        processor: HuggingFace processor for the vision-to-seq model.
        model: Pretrained HuggingFace model for vision-to-seq tasks.
        device (str): 'cuda' or 'cpu', determines where inference runs.
    """
    root_folder = Path(root_folder)
    output_root = Path(output_root)

    # Collect all image paths first
    image_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(Path(dirpath) / file)

    # Process with tqdm progress bar
    for image_path in tqdm(image_paths, desc="Processing images"):
        relative_path = image_path.parent.relative_to(root_folder)
        output_path = output_root / relative_path
        os.makedirs(output_path, exist_ok=True)

        md_output_path = output_path / image_path.with_suffix(".md").name
        image = load_image(image_path)

        try:
            markdown = process_image_to_docling_markdown(image, processor, model, device)
            with open(md_output_path, "w", encoding="utf-8") as f:
                f.write(markdown)
        except Exception as e:
            print(f"\nFailed to process {image_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert images to Docling-style markdown using SmolDocling.")
    parser.add_argument("input_folder", help="Root folder containing image files (.png, .jpg, .jpeg).")
    parser.add_argument("output_folder", help="Folder to save the generated markdown files.")
    args = parser.parse_args()

    device = get_device()
    processor, model = load_model_and_processor(device)

    process_image_folders(
        root_folder=args.input_folder,
        output_root=args.output_folder,
        processor=processor,
        model=model,
        device=device
    )


if __name__ == "__main__":
    main()
