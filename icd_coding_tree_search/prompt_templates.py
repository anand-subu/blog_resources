prompt_template_dict = {"gpt-3.5-turbo-0613" : """[Case note]:
{note}
[Example]:
<example prompt>
Gastro-esophageal reflux disease
Enteropotosis

<response>
Gastro-esophageal reflux disease: Yes, Patient was prescribed omeprazole.
Enteropotosis: No.

[Task]:
Consider each of the following ICD-10 code descriptions and evaluate if there are any related mentions in the case note.
Follow the format in the example precisely.

{code_descriptions}""",

"meta-llama/Llama-2-70b-chat-hf": """[Case note]:
{note}

[Example]:
<code descriptions>
* Gastro-esophageal reflux disease
* Enteroptosis
* Acute Nasopharyngitis [Common Cold]
</code descriptions>

<response>
* Gastro-esophageal reflux disease: Yes, Patient was prescribed omeprazole.
* Enteroptosis: No.
* Acute Nasopharyngitis [Common Cold]: No.
</response>

[Task]:
Follow the format in the example response exactly, including the entire description before your (Yes|No) judgement, followed by a newline. 
Consider each of the following ICD-10 code descriptions and evaluate if there are any related mentions in the Case note.

{code_descriptions}"""
}                   
      

