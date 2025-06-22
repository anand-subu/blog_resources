# Implementing Agentic Object Detection

## Introduction

Agentic Object Detection was touched upon by Andrew Ng [recently](https://www.linkedin.com/posts/andrewyng_introducing-agentic-object-detection-given-activity-7293302466249441280-GxAl?utm_source=share&utm_medium=member_desktop&rcm=ACoAABug4sMBXeiiqK6sYTENlHV1ZZl1T1l5neM), as part of [LandingAI](https://va.landing.ai/demo/agentic-od). The implementation is open-source and available on LandingAI's [GitHub repository](https://github.com/landing-ai/vision-agent).

This repository explores the mechanics of an agentic object detection pipeline. Traditionally, object detection has been a closed-vocabulary, closed-domain problem, where models could only detect objects they were explicitly trained on. These models lacked the flexibility to recognize objects beyond their training data. However, the emergence of multimodal models like CLIP has enabled open-vocabulary object detection models such as OWL-ViT, Florence, and DINO, which can detect objects in images based on text prompts.

While open-vocabulary models offer greater flexibility than their predecessors, they may still struggle to detect objects tied to nuanced concepts or interpret user queries accurately. Agentic Object Detection aims to bridge this gap by enabling a system to reason about a user’s request and detect the relevant concepts in the image accordingly. LandingAI [demonstrated](https://landing.ai/agentic-object-detection) that their technique leads to improved performance on object detection benchmarks compared to other existing techniques

## Pipeline

This repository attempts to build an Agentic Object Detection pipeline using Vision-Language Models (VLMs) alongside Open-Vocabulary Object Detection models. Our approach involves a 5-step pipeline, where VLMs act as reasoning agents and reviewers, working in tandem with object detection models to refine and improve the detection process.

![image](https://github.com/user-attachments/assets/8bb2f6f2-9798-4f9e-a301-19a66c49dc66)


### Our Approach:
1. Given a user-uploaded image and request, we first process them with a vision-language model (**GPT-4o**) to infer target object concepts. If the request explicitly specifies objects, we extract those. Otherwise, the VLM identifies all visible objects in the image.

2. The inferred object concepts are passed to an open-vocabulary object detection model (**Grounding Dino**) to extract bounding boxes.

3. The detected objects are visualized by padding the image with a white border, assigning arrows, and labeling each object with a unique number. This annotated image, along with the extracted concepts and user request, is then processed by a **VLM reasoner (OpenAI's o1)** to validate detections using **Chain-of-Thought**. If inaccuracies are found, the model refines object categories to higher-level abstractions **(ex: cricketers → people, poodle → dog, bats → bat)** before re-running detection.

4. The refined concepts and original image are reprocessed through the object detection model to generate updated bounding boxes.

5. The final detections are again annotated with arrows and unique numbers. This revised image, along with the user's request, is reviewed by a **VLM reasoner (GPT-4o)** using **Chain-of-Thought** to filter out irrelevant detections, retaining only those aligned with the user's request.

## Demo Video

https://github.com/user-attachments/assets/0cdb7e2d-9e15-4e53-8967-8556f9e09b0a

(Video was recorded using https://cursorful.com/ and edited using Flexclip for speeding up the video. The inference process is sped up by a higher factor for representation. The actual inference will take quite a bit of time. In  this repo, the demo video is only allowed for personal and non-commercial use to respect the terms of the Cursorful tool for free usage.)

The image used in this video is a photo by <a href="https://unsplash.com/@frank_leuderalbert?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Frank Leuderalbert</a> on <a href="https://unsplash.com/photos/blue-ceramic-cup-with-saucer-on-table-RTVYnQsLgZ0?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      )

## Prior Work
This work is meant as our interpretation and exploration of how Agentic Object Detection could work - leveraging VLMs and Open Vocab Object Detection models. There has been prior work that has explored these concepts from similar angles. Please let me know if I have missed out any relevant work.

1. [LandingAI's implementation of Agentic Object Detection](https://github.com/landing-ai/vision-agent/blob/main/vision_agent/tools/tools.py) and their [associated blog post](https://landing.ai/agentic-object-detection)
2. [DetGPT: Detect What You Need via Reasoning](https://arxiv.org/pdf/2305.14167) - This concept also leverages VLMs to assist in detecting objects with nuanced concepts.
3. [How to Fine-tune PaliGemma for Object Detection Tasks](https://blog.roboflow.com/how-to-fine-tune-paligemma/) - VLMs can theoretically be fine-tuned for object detection tasks - and this can include fine-tuning to detect objects for more nuanced user queries.

## Limitations
1. High-latency - We make a lot of VLM calls to OpenAI models (o1 and gpt-4o) that makes it slow for inference.
2. Occasional Non-Determinism in results - The initial VLM call to GPT-4o can detect different concepts each time - leading to different outputs from the object detection models for the same image across different runs.

## Contributors
Built by [Anand Subramanian](https://www.linkedin.com/in/anand-subu/) and [Bharath Sripathy](https://www.linkedin.com/in/bharath-sripathy-866666156/)

## Running the Demo
1. Run `pip install -r requirements.txt`
2. Run `python main.py` to spin up the Gradio Demo App.



