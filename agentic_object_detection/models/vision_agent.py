from PIL import Image, ImageDraw, ImageFont

from models.vlm_tool import VLMTool
from models.object_detection_tool import ObjectDetectionTool
from models.segmentation_tool import SegmentationTool
from config import DEVICE, CONFIDENCE_THRESHOLD
from utils.image_utils import encode_image

class VisionAgent:
    """
    Orchestrates calls to LLMTool, ObjectDetectionTool, and SegmentationTool,
    depending on the user’s intent.
    """
    def __init__(self, llm_api_key, model_id, obj_det_concept_extraction_model="gpt-4o", obj_det_initial_critique_model="o1", obj_det_final_critique_model="gpt-4o"):
        # 1. Initialize VLMTool
        self.vlm_tool = VLMTool(api_key=llm_api_key)

        # 2. Initialize object detection
        self.object_detection_tool = ObjectDetectionTool(
            model_id=model_id,
            device=DEVICE,
            vlm_tool=self.vlm_tool,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            concept_detection_model=obj_det_concept_extraction_model, 
            initial_critique_model=obj_det_initial_critique_model, 
            final_critique_model=obj_det_final_critique_model
        )

        # 3. Initialize segmentation
        self.segmentation_tool = SegmentationTool()

    def _handle_object_detection(self, image_path, user_text):
        return self.object_detection_tool.run(image_path, user_text)

    def _handle_semantic_segmentation(self, image_path, user_text):
        """
        1) Run object detection tool → bounding boxes
        2) Run segmentation tool on those bounding boxes
        3) Combine results
        """
        det_result = self.object_detection_tool.run(image_path, user_text)
        if det_result[0] is None:
            # Means no detection image was returned (error or nothing found)
            return det_result

        # We already have the final bounding-box image and text in `det_result`.
        # But let's also get the actual bounding boxes from object_detection_tool.last_filtered_objects.
        filtered_objects = self.object_detection_tool.last_filtered_objects
        annotated_image = det_result[0]  # PIL image with bounding boxes

        # Now segment using SAM
        segmented_overlay, masks, scores = self.segmentation_tool.run(annotated_image, filtered_objects)
        segmented_image = Image.fromarray(segmented_overlay).convert("RGB")

        final_text = det_result[1] + f"\n🔍 {len(filtered_objects)} object(s) segmented."
        return segmented_image, final_text

    def _handle_general_vlm(self, image_path, user_text, model="gpt-4o"):
        """
        Just pass the image + text to the LLM with no object detection/segmentation.
        """
        base64_image = encode_image(image_path)
        if not base64_image:
            return None, "Error processing image."

        messages = [
            {"role": "user", 
             "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
            ]}
        ]
        reply = self.llm_tool.chat_completion(messages, model=model)
        # Return no new image, just text
        return None, reply

    def process_request(self, image_path, user_text, router_model="gpt-4o"):
        """
        1. Classify user intent via LLM.
        2. Route to the relevant pipeline.
        """
        intent = self.vlm_tool.classify_intent(user_text, model=router_model)
        if intent == "object_detection":
            return self._handle_object_detection(image_path, user_text)
        elif intent == "semantic_segmentation":
            return self._handle_semantic_segmentation(image_path, user_text)
        elif intent == "general_vlm":
            return self._handle_general_vlm(image_path, user_text)
        else:
            return NotImplementedError("This function is not implemented yet")