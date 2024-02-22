## Setting up the environment

This code was tested in a python 3.9 runtime. 
Run ```pip install -r requirements.txt``` to download the required packages.

## Running the code

1. You will need an OpenAI account with an API key for processing the data with the GPT-3.5 API.
2. You can set up the API key in your environment variable.
3. The ```run_inference_llama_chat.ipynb``` contains all the code for running inference with the Llama-2 model. You'll need to obtain access to the Llama-2 model to download it.
4. The ```run_inference_gpt.ipynb``` contains code for processing the dataset using GPT-3.5.
5. All outputs are dumped after running inference.
6. The subsampled test set and few-shot prompts are in the artefacts folder. They can be moved to the same folder as the notebook for running the code.
