import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np

from utils.image_utils import generate_overlay_image, load_image, bboxes_to_points

class SegmentationTool:
    """
    Segments bounding boxes using SAM.
    """
    def __init__(self):
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    def run(self, image, filtered_objects):
        """
        Takes an image (PIL) + a list of (number, label, bbox).
        Returns an overlayed segmentation image and any text you want.
        """
        bboxes = [item[-1] for item in filtered_objects]  # (n, label, [x1,y1,x2,y2])
        input_points = bboxes_to_points(bboxes)

        all_masks = []
        all_scores = []
        image_pil = load_image(image)

        for points in input_points:
            inputs = self.processor(
                image_pil,
                input_points=[points],
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )
            scores = outputs.iou_scores

            # Take the best mask
            if len(scores.shape) == 3:
                best_mask_idx = scores[0, 0].argmax()
                mask = masks[0][0].numpy()
                score = scores[0, 0, best_mask_idx].item()
            else:
                best_mask_idx = scores[0].argmax()
                mask = masks[0][0].numpy()
                score = float(scores[0].max())

            all_masks.append(mask)
            all_scores.append(score)

        overlay_img = generate_overlay_image(image_pil, all_masks)
        return overlay_img, all_masks, all_scores