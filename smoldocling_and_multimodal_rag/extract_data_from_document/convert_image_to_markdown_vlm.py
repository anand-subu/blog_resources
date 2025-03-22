import os
import base64
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


def encode_image(image_path):
    """
    Reads an image file from the given path and encodes its contents in base64 format.

    Args:
        image_path (str or Path): The path to the image file.

    Returns:
        str: The base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def analyze_image(image_path, client, model, max_tokens, temperature):
    """
    Sends a base64-encoded image to the OpenAI API and retrieves a Markdown description of its visible content.

    Args:
        image_path (Path): The path to the image file.
        client: An instance of the OpenAI API client.
        model (str): The model name to use for the API request.
        max_tokens (int): The maximum number of tokens allowed in the response.
        temperature (float): The sampling temperature for response randomness.

    Returns:
        str: A Markdown-formatted description of the image content.
    """
    base64_image = encode_image(image_path)

    # Determine MIME type based on extension
    ext = image_path.suffix.lower()
    if ext in {'.jpg', '.jpeg'}:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        mime_type = 'image/jpeg'  # default fallback

    messages = [
        {"role": "system", "content": "You are a helpful assistant that describes images in Markdown format."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all visible content from the image and present it in clean, structured Markdown. "
                            "Include accurate captions for any images, figures, or diagrams. Do not add any content that is not present in the  image. Output only the Markdown directly—no explanations or extra text or any markdown code block identifiers."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
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


def get_all_image_paths(root, extensions):
    """
    Recursively retrieves all image file paths within a directory that match the specified extensions.

    Args:
        root (str or Path): The root directory to search.
        extensions (set): A set of valid image file extensions (e.g., {".jpg", ".png"}).

    Returns:
        list[Path]: A list of Paths to image files with matching extensions.
    """
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in extensions]


def get_output_path(image_path, input_root, output_root):
    """
    Constructs the output Markdown file path that mirrors the input image's directory structure.

    Args:
        image_path (Path): The full path to the image file.
        input_root (Path): The root directory of the input images.
        output_root (Path): The root directory where Markdown files will be saved.

    Returns:
        Path: The full path to the corresponding output Markdown file.
    """
    relative_path = image_path.relative_to(input_root)
    output_path = Path(output_root) / relative_path.parent / (image_path.stem + ".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def process_image(image_path, input_root, output_root, client, model, max_tokens, temperature):
    """
    Processes a single image by analyzing its content and writing the result as a Markdown file.

    Args:
        image_path (Path): The path to the image file.
        input_root (Path): The root input directory (used to compute relative paths).
        output_root (Path): The root output directory where Markdown files are saved.
        client: An instance of the OpenAI API client.
        model (str): The model name to use for the API request.
        max_tokens (int): The maximum number of tokens allowed in the API response.
        temperature (float): The sampling temperature for response generation.

    Returns:
        tuple: A tuple of (image_path, status message) indicating success or failure.
    """
    try:
        markdown = analyze_image(image_path, client, model, max_tokens, temperature)
        output_path = get_output_path(image_path, input_root, output_root)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return (image_path, "Success")
    except Exception as e:
        return (image_path, f"Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown descriptions from images using GPT-4o-mini.")
    parser.add_argument("--input_dir", help="Directory containing images.")
    parser.add_argument("--output_dir", help="Directory to save markdown output.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use. Default: gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=3000, help="Max tokens for OpenAI response. Default: 3000")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation. Default: 0.0")
    parser.add_argument("--max-workers", type=int, default=5, help="Max parallel threads. Default: 5")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or args.api_key
    if not api_key:
        raise ValueError("OpenAI API key is required. Provide it via the OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_paths = get_all_image_paths(input_root, extensions)
    print(f"Found {len(image_paths)} images.")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_image, img_path, input_root, output_root,
                client, args.model, args.max_tokens, args.temperature
            ): img_path
            for img_path in image_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            img_path, status = future.result()
            results.append((img_path, status))

    # Summary
    print("\nProcessing Summary:")
    for img_path, status in results:
        print(f"{img_path}: {status}")


if __name__ == "__main__":
    main()
