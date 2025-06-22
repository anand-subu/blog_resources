import cv2
import base64
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def encode_image(image_path):
    """
    Encodes an image file into a base64-encoded string.

    Args:
        image_path (str): Path to the image file to be encoded.

    Returns:
        str: Base64-encoded string of the image content.
        None: If an error occurs during encoding.

    Raises:
        Exception: If the file cannot be opened or encoded.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def draw_arrows_and_numbers(image_path, detected_objects):
    """
    Draws arrows and numbers on an image to label detected objects.

    This function dynamically places numbers near the borders with arrows pointing 
    from object centers to the borders. Arrows are dashed for clarity, and the 
    numbering avoids overlap when possible.

    Args:
        image_path (str): Path to the input image.
        detected_objects (list): List of tuples containing object information in 
            the format (number, object_name, bounding_box), where bounding_box is 
            (x1, y1, x2, y2).

    Returns:
        str: Path to the saved labeled image ('labeled_objects_optimized.jpg').

    Note:
        - Arrows are drawn from object centers to the nearest border.
        - Numbers are displayed with semi-transparent backgrounds for readability.
    """
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    used_positions = []

    # Pad the image with a white border
    top, bottom, left, right = 50, 50, 50, 50
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    height, width, _ = img.shape

    for i, (num, obj, box) in enumerate(detected_objects):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box

        # Adjust coordinates for padded image
        x1 += left
        y1 += top
        x2 += left
        y2 += top
        cx += left
        cy += top

        # Determine arrow direction towards the nearest border
        distances = {'top': cy, 'bottom': height - cy, 'left': cx, 'right': width - cx}
        direction = min(distances, key=distances.get)

        if direction == 'top':
            arrow_end = (cx, top)
            text_position = (cx - 10, top - 10)
        elif direction == 'bottom':
            arrow_end = (cx, height - bottom)
            text_position = (cx - 10, height - 5)
        elif direction == 'left':
            arrow_end = (left, cy)
            text_position = (left - 30, cy + 5)
        else:
            arrow_end = (width - right, cy)
            text_position = (width - 30, cy + 5)

        # Draw the dashed arrow from the object center to the border
        color = (0, 0, 0)  # Black color for all arrows
        line_type = cv2.LINE_4
        cv2.arrowedLine(img, (cx, cy), arrow_end, color, 2, tipLength=0.3)

        # Draw a semi-transparent rectangle behind the text
        overlay = img.copy()
        cv2.rectangle(overlay, (text_position[0] - 5, text_position[1] - 20), (text_position[0] + 30, text_position[1] + 5), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw the number at the border with black text
        cv2.putText(img, str(num), text_position, font, 0.8, color, 2)

    labeled_image_path = "labeled_objects_optimized.jpg"
    cv2.imwrite(labeled_image_path, img)
    return labeled_image_path

    
def load_image(image_input):
    """
    Loads an image from a file path, PIL Image, or NumPy array.

    Args:
        image_input (str | Image.Image | np.ndarray): Input image in one of the supported formats.

    Returns:
        Image.Image: The loaded image in RGB mode.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the input type is unsupported.
    """
    if isinstance(image_input, str):  # Path
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        return Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):  # PIL
        return image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):  # Numpy
        return Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError("Unsupported image input type. Please provide a file path, PIL Image, or numpy array")

def bboxes_to_points(bboxes):
    """
    Converts a list of bounding boxes into center-point prompts for SAM.

    Args:
        bboxes (list of tuples): Each tuple should be in the format (x1, y1, x2, y2).

    Returns:
        list: A list of center points in the format [[center_x, center_y]] for each bounding box.

    Note:
        The center point is calculated as the midpoint of the bounding box.
    """
    all_points = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        all_points.append([[center_x, center_y]])  # each bounding box → one center point
    return all_points

def generate_overlay_image(image, masks, alpha=0.3):
    """
    Generates an overlay image by blending segmentation masks with the original image.

    Args:
        image (Image.Image | np.ndarray): The original image to overlay masks onto.
        masks (list of np.ndarray): List of segmentation masks. Each mask is expected 
            to be binary with the same dimensions as the image.
        alpha (float, optional): Transparency factor for the overlay color. Default is 0.3.

    Returns:
        np.ndarray: The image with segmentation masks overlaid.

    Note:
        - Masked areas are highlighted in a semi-transparent red color.
        - If the input image is a PIL Image, it is converted to a NumPy array before processing.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    overlay = image.copy()
    for mask in masks:
        if len(mask.shape) == 3:
            mask = mask[0]
        binary_mask = mask > 0.5
        overlay[binary_mask] = overlay[binary_mask] * 0.7 + np.array([255, 0, 0]) * 0.3

    return overlay

def draw_rounded_rectangle(draw, box, radius, outline, width):
    """
    Draws a rounded rectangle on an image.

    This function creates a rectangle with rounded corners by drawing straight lines 
    and pieslices for the corners.

    Args:
        draw (ImageDraw.Draw): The ImageDraw object used to draw on the image.
        box (tuple): Coordinates of the rectangle as (x1, y1, x2, y2).
        radius (int): The radius of the rounded corners.
        outline (str or tuple): The color of the rectangle outline.
        width (int): The width of the rectangle outline.

    Notes:
        - The corners are drawn as quarter circles using the `pieslice` method.
        - The rectangle is drawn by connecting these corners with lines.
    """
    x1, y1, x2, y2 = box
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], outline=outline, width=width)

def draw_bounding_boxes(image_path, filtered_objects):
    """
    Draws bounding boxes with labels on an image.

    For each object in the `filtered_objects` list, this function draws a bounding box 
    with rounded corners and labels it with the corresponding object name.

    Args:
        image_path (str): Path to the input image file.
        filtered_objects (list of tuples): Each tuple contains (number, label, box) where:
            - number (int): Identifier or index for the object.
            - label (str): Label for the detected object.
            - box (tuple): Bounding box coordinates as (x1, y1, x2, y2).

    Returns:
        Image.Image: The image with bounding boxes and labels drawn.

    Notes:
        - Bounding boxes have rounded corners.
        - Labels are displayed in a semi-transparent box above each bounding box.
        - The font is assumed to be `arial.ttf` and should exist in the working directory.
    """
    # Open the image
    final_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(final_img)
    font = ImageFont.truetype("arial.ttf", 20)

    for num, lbl, box in filtered_objects:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1

        # Define styles
        outline_color = "red"
        outline_width = 3
        radius = 10  # Radius for rounded corners
        label_fill_color = (255, 0, 0, 128)  # Semi-transparent red
        text_color = "white"
        padding = 5

        # Draw rounded rectangle for bounding box
        draw_rounded_rectangle(draw, [x1, y1, x2, y2], radius, outline_color, outline_width)

        # Calculate text size
        text_size = draw.textsize(lbl, font=font)
        text_width, text_height = text_size

        # Position for label box
        label_box = [x1, y1 - text_height - 2 * padding, x1 + text_width + 2 * padding, y1]

        # Draw label box
        draw.rectangle(label_box, fill=label_fill_color)

        # Draw text
        text_position = (x1 + padding, y1 - text_height - padding)
        draw.text(text_position, lbl, fill=text_color, font=font)

    return final_img
