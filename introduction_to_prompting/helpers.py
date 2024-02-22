import json
import re
from openai import OpenAI

pattern = re.compile(r"([A-Z])[.:]\s*(.*)")

PROMPT = """You will be provided with a medical or clinical question, along with multiple possible answer choices. Pick the right answer from the choices. 
Your response should be in the format "The answer is <correct_choice>". Do not add any other unnecessary content in your response"""

COT_INSTRUCTION = """Instructions: The following are multiple-choice questions about medical knowledge. Solve them in a step-by-step fashion.
Output a single option as the final answer."""

QUESTION_1 = {"question": """Question: A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
(A) anterior scalene (B) latissimus dorsi (C) pectoralis minor (D) quadratus lumborum""",
             "explanation": """Explanation: We refer to Wikipedia articles on medicine for help. Among the options, only pectoralis minor muscle origins from the outer surfaces of the 3rd to 5th ribs.
Answer: (C)"""}

QUESTION_2 = {"question": """Question: A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
(A) left-on-left sacral torsion (B) left-on-right sacral torsion (C) right unilateral sacral flexion (D) right-on-right sacral torsion""",
             "explanation": """Explanation: We refer to Wikipedia articles on medicine for help. The deep sulcus on the left, a posterior ILA on the right, with a negative spring test suggests a right-on-right sacral torsion. All other options have a deep sulcus on the right.
Answer: (D)"""}

QUESTION_3 = {"question": """Question: A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient’s symptoms?
(A) Allergic rhinitis (B) Epstein-Barr virus (C) Mycoplasma pneumonia (D) Rhinovirus""",
             "explanation": """Explanation: We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it’s not Mycoplasma pneumonia.
Answer: (D)"""}


QUESTION_4 = {"question": """Question: A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
(A) Dopamine (B) Glutamate (C) Norepinephrine (D) Serotonin""",
             "explanation": """Explanation: We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient’s decreased appetite and sleep difficulty.
Answer: (D)"""}


QUESTION_5 = {"question": """Question: A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
(A) Labetalol (B) A loading dose of potassium chloride (C) Nifedipine (D) Phenoxybenzamine""",
             "explanation": """Explanation: We refer to Wikipedia articles on medicine for help. The symptoms and the adrenal mass suggested pheochromocytoma, and the blood pressure indicates hypertension. Phenoxybenzamine is used to treat hypertension caused by pheochromocytoma.
Answer: (D)"""}

COT_EXAMPLES = [QUESTION_1, QUESTION_2, QUESTION_3, QUESTION_4, QUESTION_5]


client = OpenAI()

def parse_answer(response):
    """
    Extracts the answer option from the predicted string.

    Args:
    - response (str): The string to search for the pattern.

    Returns:
    - str: The matched answer option if found or an empty string otherwise.
    """
    match = re.search(pattern, response)
    if match:
        letter = match.group(1)
    else:
        letter = ""
    
    return letter

def calculate_accuracy(ground_truth, predictions):
    """
    Calculates the accuracy of predictions compared to ground truth labels.

    Args:
    - ground_truth (list): A list of true labels.
    - predictions (list): A list of predicted labels.

    Returns:
    - float: The accuracy of predictions as a fraction of correct predictions over total predictions.
    """
    return sum([1 if x==y else 0 for x,y in zip(ground_truth, predictions)]) / len(ground_truth)


def read_jsonl_file(file_path):
    """
    Parses a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list of dict: A list where each element is a dictionary representing
            a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

def write_jsonl_file(file_path, dict_list):
    """
    Write a list of dictionaries to a JSON Lines file.

    Args:
    - file_path (str): The path to the file where the data will be written.
    - dict_list (list): A list of dictionaries to write to the file.
    """
    with open(file_path, 'w') as file:
        for dictionary in dict_list:
            # Convert the dictionary to a JSON string and write it to the file.
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def create_query(item):
    """
    Creates the input for the model using the question and the multiple choice options.

    Args:
        item (dict): A dictionary containing the question and options.
            Expected keys are "question" and "options", where "options" is another
            dictionary with keys "A", "B", "C", and "D".

    Returns:
        str: A formatted query combining the question and options, ready for use.
    """
    query = item["question"] + "\nOptions:\n" + \
            "A. " + item["options"]["A"] + "\n" + \
            "B. " + item["options"]["B"] + "\n" + \
            "C. " + item["options"]["C"] + "\n" + \
            "D. " + item["options"]["D"]
    return query

def build_zero_shot_prompt(system_prompt, question):
    """
    Builds the zero-shot prompt.

    Args:
        content (dict): The content for which to create a query, formatted as
            required by `create_query`.

    Returns:
        list of dict: A list of messages, including a system message defining
            the task and a user message with the input question.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": create_query(question)}]
    return messages


def build_few_shot_prompt(system_prompt, content, few_shot_examples):
    """
    Builds the few-shot prompt for the GPT API using provided examples.

    Args:
        content (dict): The content for which to create a query, similar to the
            structure required by `create_query`.
        few_shot_examples (list of dict): Examples to simulate a hypothetical
            conversation. Each dict must have "options" and an "answer".

    Returns:
        list of dict: A list of messages, simulating a conversation with
            few-shot examples, followed by the current user query.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for item in few_shot_examples:
        ans_options = item["options"]
        correct_ans_option = ""
        for key, value in ans_options.items():
            if value == item["answer"]:
                correct_ans_option = key
                break
        messages.append({"role": "user", "content": create_query(item)})
        messages.append({"role": "assistant", "content": "The answer is " + correct_ans_option + "."})
    messages.append({"role": "user", "content": create_query(content)})
    return messages

def create_query_cot(item):
    """
    Creates the input for the model using the question and the multiple choice options in the CoT format.

    Args:
        item (dict): A dictionary containing the question and options.
            Expected keys are "question" and "options", where "options" is another
            dictionary with keys "A", "B", "C", and "D".

    Returns:
        str: A formatted query combining the question and options, ready for use.
    """
    query = "Question: " + item["question"] + "\n" + \
            "(A) " + item["options"]["A"] + " " +  \
            "(B) " + item["options"]["B"] + " " +  \
            "(C) " + item["options"]["C"] + " " +  \
            "(D) " + item["options"]["D"]
    return query

def build_cot_prompt(instruction, input_question, cot_examples):
    """
    Builds the few-shot prompt for the GPT API using provided examples.

    Parameters:
        content (dict): The content for which to create a query, similar to the
            structure required by `create_query`.
        few_shot_examples (list of dict): Examples to simulate a hypothetical
            conversation. Each dict must have "question" and an "explanation".

    Returns:
        list of dict: A list of messages, simulating a conversation with
            few-shot examples, followed by the current user query.
    """
    
    messages = [{"role": "system", "content": instruction}]
    for item in cot_examples:
        messages.append({"role": "user", "content": item["question"]})
        messages.append({"role": "assistant", "content": item["explanation"]})

    
    messages.append({"role": "user", "content": create_query_cot(input_question)})
    
    return messages

def parse_answer_cot(text):
    """
    Extracts the choice from a string that follows the pattern "Answer: (Choice) Text".

    Parameters:
    - text (str): The input string from which to extract the choice.

    Returns:
    - str: The extracted choice or a message indicating no match was found.
    """
    # Regex pattern to match the answer part
    pattern = r"Answer: (.*)"

    # Search for the pattern in the text and extract the matching group
    match = re.search(pattern, text)
    
    if match:
        if len(match.group(1)) > 1:
            return match.group(1)[1]
        else:
            return ""
    else:
        return ""

def get_response(messages, model_name, temperature = 0.0, max_tokens = 10):
    """
    Obtains the responses/answers of the model through the chat-completions API.

    Parameters:
        messages (list of dict): The built messages provided to the API.
        model_name (str): Name of the model to access through the API
        temperature (float): A value between 0 and 1 that controls the randomness of the output.
        A temperature value of 0 ideally makes the model pick the most likely token, making the outputs deterministic.
        max_tokens (int): Maximum number of tokens that the model should generate

    Returns:
        str: The response message content from the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content