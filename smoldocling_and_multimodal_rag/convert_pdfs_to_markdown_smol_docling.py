import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

# Set device to GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize processor and model
processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForVision2Seq.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    torch_dtype=torch.bfloat16,
).to(DEVICE)


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
    Walks through nested folders of PNG images, converts each image to Docling Markdown,
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

    for dirpath, _, filenames in os.walk(root_folder):
        image_files = [f for f in filenames if f.lower().endswith(".png")]
        if not image_files:
            continue

        relative_path = Path(dirpath).relative_to(root_folder)
        output_path = output_root / relative_path
        os.makedirs(output_path, exist_ok=True)

        for image_file in sorted(image_files):
            image_path = os.path.join(dirpath, image_file)
            print(f"Processing image: {image_path}")

            md_output_path = output_path / Path(os.path.splitext(image_file)[0] + ".md")
            image = load_image(image_path)

            try:
                markdown = process_image_to_docling_markdown(image, processor, model, device)
                with open(md_output_path, "w", encoding="utf-8") as f:
                    f.write(markdown)
                print(f"Saved markdown to: {md_output_path}")
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")


# Example usage
if __name__ == "__main__":
    process_image_folders(
        root_folder="output_images",         # Folder with subfolders of images
        output_root="output_markdown",       # Folder to save the markdown files
        processor=processor,
        model=model,
        device=DEVICE
    )
