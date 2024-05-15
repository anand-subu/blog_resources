# automated-icd-coding-llm

### Overview
This is an unofficial implementation of the algorithm described in the paper [Automated clinical coding using off-the-shelf large language models](https://openreview.net/pdf?id=mqnR8rGWkn).

### LLM Guided Tree-Search Algorithm

This paper discusses the feasibility of employing off-the-shelf Large Language Models (LLMs) for ICD coding. The authors utilize a novel and practical approach which they term LLM-guided tree-search. Each ICD code is part of a hierarchical relationship where parent codes encompass broader conditions and child codes represent specific ailments. The search begins at the root and uses the LLM to choose which branches to investigate, proceeding iteratively until no further paths remain. This process determines the most relevant ICD codes, adding them as predicted labels for the clinical note.

#### Process
The search process operates as follows:
1. **Initiation**: Begins at the ontology's root.
2. **Navigation**: Uses the LLM to determine which branches to explore.
3. **Iteration**: Proceeds iteratively, exploring further until no viable paths remain.
4. **Conclusion**: Identifies the most relevant ICD codes and adds them as predicted labels for the medical note.

### Running the code

#### Download the CodiEsp Dataset
You can download the CodiEsp Dataset from this [link](https://zenodo.org/records/3837305#.XsZFoXUzZpg)

#### Configure the OpenAI credentials for the client
* In order to run this code, you need to configure the credentials for accessing the LLMs via API. For this, you need API keys for GPT-3.5/GPT-4 and Llama-2 70B Chat.

* For GPT-3.5/GPT-4, you need to create an account with OpenAI and create an API key.

* For Llama-2 70B Chat, you can create an account with [deepinfra](https://deepinfra.com/). You can create an API key, and then create an deployment of Llama-2 70B Chat, and obtain the base url for this deployment.

* Both LLMs can be accessed through the `openai` library, and Line 9 in `helpers.py` need to be configured as follows:

##### For GPT-3.5/GPT-4
```
client = OpenAI(api_key=<OPEN_AI_API_KEY>)
```
##### For Llama-2 70B Chat
```
client = OpenAI(api_key=<DEEP_INFRA_API_KEY>,  base_url=<DEEP_INFRA_DEPLOYMENT_URL>)
```

#### Translate the Spanish medical notes to English
Once you've downloaded the dataset, run 

```
python translate_files.py --input_dir <input_dir_with_spanish_test_set> --output_dir <output_dir_to_save_files_to>
```

The script will use GPT-3.5 to translate the files and then save the outputs to the directory.

#### Run the Tree Search Algorithm
After translating the dataset, run

```
python run_tree_search.py --input_dir <translated_en_test_set_dir> --output_file <output_json_to_save_results> --model_name <model_name>
```

The model name can be either `gpt-3.5-turbo-0613` for GPT-3.5 or `meta-llama/Llama-2-70b-chat-hf` for Llama-2 70B Chat. The output predictions are dumped as a json file.

#### Evaluate the performance
The performance is evaluated in terms of macro-average and micro-average precision, recall and f1-scores.
The script for evaluation was provided by the authors of the [paper](https://openreview.net/pdf?id=mqnR8rGWkn). The evaluation script provided by the authors, is a modified version of the CodiEsp Shared Task Evaluation script.
Run the script as:
```
python evaluate_performace.py --input_json <path_to_predictions_json_file> --gold_standard_tsv <path_to_gold_standard_test_tsv>
```
Please note the script internally converts the predictions_json file to a tsv file format, and also modifies the gold standard tsv file by adding the appropriate column names and dumps the modified version locally. This is then removed by the script after evaluation is complete.

### Differences in Implementation from the original paper
**Please note** this is purely my implementation based on my understanding of the paper, and may differ in some areas compared to the original implementation, which may impact the results obtained by running this.
I've jotted down a few potential differences:

1. **Prompt used for translating the documents from Spanish to English**: The paper mentions that they translate the Spanish documents to English using GPT-3.5, instead of utilizing the original translated version. I decided to implement this as well, and translated the documents to English. However, there may be some potential differences in the translated document content based on the prompt utilized, which may have an impact on the scores. I've uploaded my version of the [translated CodiEsp test set](https://drive.google.com/file/d/1iIhtAbqmEq3MRPJMBYnAeDJTiZl-q2Nl/view?usp=sharing) for usage.

2. **Extracting the predicted code descriptions from the LLM's predictions**: In the original paper, the authors follow the following steps to extract the matched ICD Descriptions:

> Resolving the LLM generated text into per-code predictions is performed by processing the text as a set of lines. These lines are greedily matched, starting with the longest code description in the current prompt

In my implementation, I simplify the extraction logic a bit, by assuming that each predicted ICD code descriptions is on a per-line basis and split each line by ":". This is because ideally the predictions should be of the format "<ICD Description: <yes/no> ...", and splitting by the first colon should give the description and the LLM prediction. I drop the predicted line if the description extracted by the LLM is not an exact match against the ICD tree. This may have an impact on the final performance.

While I've highlighted the main differences in my implementation, there may be some other minor differences that could also impact performance. Please keep these in mind, if you try this code.

* While I've used the versions of the GPT-3.5 model with the parameters mentioned in the paper for reproducibility, it is possible that there could be some randomness inherent to the LLM calls for various reasons, that can impact the final performance.
* I've reconstructed the prompts based on the examples based in the paper. There could be some minor differences in the prompt that may affect the performance.

### Results
The results of the evaluations are presented here. This was possible, thanks to the authors of the paper who shared their evaluation code when I communicated with them. 
However, while my implementation's metrics are roughly in the ball-park  of the reported scores, there are some note-worthy differences:
1. In my implementation, GPT-3.5's micro-average metrics slightly exceed the reported figures, whereas the macro-average metrics fall a bit short of the reported values.
2. In my implementation, Llama-70B's micro-average metrics either match or slightly exceed the reported figures, but the macro-average metrics are lesser than the reported values.

As mentioned earlier, this implementation differs from the paper in some small ways, all of which impact the final performance.

| Model     | Micro-Average Precision | Micro-Average Recall | Micro-Average F1-Score | Macro-Average Precision | Macro-Average Recall | Macro-Average F1-Score |
|-----------|:-----------------------:|:--------------------:|:----------------------:|:-----------------------:|:--------------------:|:----------------------:|
| GPT-3.5   |           0.173         |        0.241         |        0.201           |           0.219         |         0.213        |        0.196           |
| Llama-2 70B Chat|           0.051         |        0.172         |        0.078           |           0.113         |         0.155        |         0.11           |

### Citation
If you use this code, please cite the original papers:
```
@inproceedings{
boyle2023automated,
title={Automated clinical coding using off-the-shelf large language models},
author={Joseph Boyle and Antanas Kascenas and Pat Lok and Maria Liakata and Alison O'Neil},
booktitle={Deep Generative Models for Health Workshop NeurIPS 2023},
year={2023},
url={https://openreview.net/forum?id=mqnR8rGWkn}
}
```

If you use the CodiEsp Dataset, please cite the following papers:

```
@inproceedings{
miranda2020overview,
title={Overview of automatic clinical coding: annotations, guidelines, and solutions for non-english clinical cases at codiesp track of CLEF eHealth 2020},
author={Miranda-Escalada, Antonio and Gonzalez-Agirre, Aitor and Armengol-Estap{'e}, Jordi and Krallinger, Martin},
booktitle={Working Notes of Conference and Labs of the Evaluation (CLEF) Forum. CEUR Workshop Proceedings},
year={2020}
}
```

```
@dataset{miranda_escalada_2020_3837305,
  author       = {Miranda-Escalada, Antonio and
                  Gonzalez-Agirre, Aitor and
                  Krallinger, Martin},
  title        = {{CodiEsp corpus: gold standard Spanish clinical 
                   cases coded in ICD10 (CIE10) - eHealth CLEF2020}},
  month        = may,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.4},
  doi          = {10.5281/zenodo.3837305},
  url          = {https://doi.org/10.5281/zenodo.3837305}
}
```
