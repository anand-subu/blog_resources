### Usage
1. Run `pip install -r requirements.txt`

2. Run the script to generate testsets from markdown documents in subfolders:

```bash
python generate_testsets.py --base-dir <path_to_markdown_extracted_files> --output-dir <path_to_store_generated_testsets> --testset-size <num_of_questions_to_generate_perfolder> --model <open_ai_llm_to_use>

