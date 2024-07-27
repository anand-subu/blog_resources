## Understanding and Implementing Medprompt [[Blog Article](https://medium.com/towards-data-science/understanding-and-implementing-medprompt-77bbd2777c91)]
This directory contains the code and resources for running the experiments described in the blog article.

## Running the code

1. You will need an OpenAI account with an API key for processing the data with the GPT-4o API.
2. The ```medprompt.ipynb``` contains all the code for running the Medprompt method.
3. All relevant artifacts and resources for this experiment can be found at this [link]([https://drive.google.com/drive/folders/16itn9D7RMD_sXfTElNANspxrWDZ_Bok7?usp=sharing]).
4. If you want to run the code from scratch, you can download the MedQA dataset from this [repo] (https://github.com/jind11/MedQA)
5. I've provided the intermediate artifacts and dumped them as jsonl files, if you want to directly skip some steps.
6. ```test_data_usmle_subsampled.jsonl``` in the drive link contains a subsampled version of the full test set that I used for my experiment.
7. ```final_processed_test_set_responses_medprompt.jsonl``` in the drive link contains the final processed outputs from Medprompt for the test set based on my experiment.
8. ```cot_responses_medqa_train_set_filtered_with_embeddings.jsonl``` in the drive link contains the processed versions of the train set with Self-Generated CoT outputs and embeddings.
