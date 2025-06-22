import base64
import json
from openai import OpenAI

from utils.image_utils import encode_image

class VLMTool:
    """
    Handles LLM calls (e.g. GPT-4).
    """
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, messages, model="o1", max_tokens=300, temperature=0.1, response_format=None):
        """Calls GPT for chat completion."""
        try:
            
            if model in ["gpt-4o", "gpt-4o-mini"]:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format if response_format else {"type": "text"}
                )
            elif model in ["o1"]:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format if response_format else {"type": "text"}
                )
            
            else:
                raise NotImplementedError("This model is not supported")

            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

    def classify_intent(self, text, model="gpt-4o"):
        """Uses GPT to classify user intent: 'object_detection', 'semantic_segmentation', or 'general_vlm'."""

        messages = [
            {"role": "system", 
             "content": "You are an AI assistant that classifies user intent for vision tasks. "
                        "Keep in mind that you only need to classify the nature of the task based on the user's request."
                        "Only reply with 'object_detection', 'semantic_segmentation', or 'general_vlm'."},
            {"role": "user", 
             "content": [
                {"type": "text", "text": text},
            ]}
        ]
        
        return self.chat_completion(messages, model=model).strip().lower()

    def extract_objects_from_request(self, image_path, user_text, model="gpt-4o"):
        """
        Asks the LLM to parse user request for which objects to detect/segment.
        Returns a list of objects in plain text.
        """
        base64_image = encode_image(image_path)
        if not base64_image:
            return None

        prompt = (
        "You are an AI vision assistant that extracts objects to be identified from a user's request."
        "If the user wants to detect or semantically segment all objects in the image, return a comma-separated list of objects you can see. "
        "If the user wants to detect or semantically segment specific objects, extract only those mentioned explicitly in their request. "
        "Respond ONLY with the list of objects, separated by commas, and NOTHING ELSE."
        "The objective here is only to understand the objects of interest that can be extracted from the image and the user's request."
        "You are not actually required to perform or execute the user's request."    
            
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", 
                 "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
            ]}
        ]

        result = self.chat_completion(messages, model=model)
        if result:
            detected_objects = [obj.strip().lower() for obj in result.split(",") if obj.strip()]
            return detected_objects
        return []
