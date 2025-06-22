import gradio as gr
from models.vision_agent import VisionAgent
from config import OPENAI_API_KEY, MODEL_ID

# Instantiate the agent
agent = VisionAgent(llm_api_key=OPENAI_API_KEY, model_id=MODEL_ID)

def gradio_interface(image, text):
    image_path = "temp_input.jpg"
    image.save(image_path)
    result = agent.process_request(image_path, text)
    if not result:
        return image, "Error"
    out_img, out_text = result
    return (image if out_img is None else out_img), out_text

# Gradio interface
inputs = [gr.Image(type="pil", label="Uploaded Image"), gr.Textbox(lines=2, label="User Request", placeholder="Enter your request...")]
outputs = [gr.Image(type="pil", label="Processed Image"), gr.Textbox(label="Processed Text")]

gr.Interface(
    fn=gradio_interface, 
    inputs=inputs, 
    outputs=outputs, 
    title="Vision Agent"
).launch()
