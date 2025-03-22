import base64
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Configuration
api_key = ""
client = OpenAI(api_key=api_key)

input_root = "output_images/"
output_root = "markdown_gpt4o_mini"
model = "gpt-4o-mini"
max_tokens = 3000
temperature = 0.0
max_workers = 5


def encode_image(image_path):
    """
    Reads an image file and encodes it in base64 format.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64-encoded string of the image content.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def analyze_image(image_path):
    """
    Sends an image to the OpenAI API and retrieves a Markdown description.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Markdown-formatted description of the image content.
    """
    base64_image = encode_image(image_path)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that describes images in Markdown format."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all visible content from the image and present it in clean, structured Markdown. Include accurate captions for any images, figures, or diagrams. Output only the Markdown—no explanations or extra text."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content


def get_all_image_paths(root, extensions={".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}):
    """
    Recursively retrieves all image file paths from a given directory.

    Args:
        root (str or Path): Root directory to search.
        extensions (set): Allowed image file extensions.

    Returns:
        list: List of Path objects for each image file.
    """
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in extensions]


def get_output_path(image_path):
    """
    Computes the corresponding output path for the Markdown file.

    Args:
        image_path (Path): Original image file path.

    Returns:
        Path: Output path for the Markdown file.
    """
    relative_path = image_path.relative_to(input_root)
    output_path = Path(output_root) / relative_path.parent / (image_path.stem + ".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def process_image(image_path):
    """
    Processes a single image and writes its Markdown description to disk.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        tuple: (image_path, status message)
    """
    try:
        markdown = analyze_image(image_path)
        output_path = get_output_path(image_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return (image_path, "Success")
    except Exception as e:
        return (image_path, f"Failed: {e}")


def main():
    """
    Main execution function that processes all images in the input directory.
    """
    image_paths = get_all_image_paths(input_root)
    print(f"Found {len(image_paths)} images.")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, img_path): img_path for img_path in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            img_path, status = future.result()
            results.append((img_path, status))

    # Summary
    print("\nProcessing Summary:")
    for img_path, status in results:
        print(f"{img_path}: {status}")


if __name__ == "__main__":
    main()
