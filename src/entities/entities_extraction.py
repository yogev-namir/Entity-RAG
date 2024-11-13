import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from config import RELEVANT_CATEGORIES, PREDEFINED_SEX_MAPPING
import re

load_dotenv()
ner_model_name = os.getenv('NER_MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)


def extract_entities(text: str, id: int, partial=True, train=True):
    """
    1. Tokenize the input
    2. Get the model output
    3. Extract the logits (raw predictions) and convert them into predicted class IDs
    4. Map predicted IDs to actual entity labels using the tokenizer's label mapping,
    5. Get the mapping of class ids to label names
    6. Initialize an empty dictionary to store results and declare variables to keep track of the current entity,
    7. Process the tokens and their labels
    :param train: True if input is part of train set, False for test queries
    :param text: text to extract entities
    :param id: chunk ID
    :param partial:
    :return:
    """

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_ids = torch.argmax(logits, dim=2)

    label_ids = model.config.id2label
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    entities = {}
    current_entity = None
    current_label = None

    for token, predicted_id in zip(tokens, predicted_class_ids[0]):
        label = label_ids[predicted_id.item()]
        token = token.lower()

        if label.startswith("B-"):  # Beginning of a new entity
            # Save the previous entity
            if current_entity and current_label:
                if current_label not in entities:
                    entities[current_label] = []
                entities[current_label].append(current_entity)

            # Start a new entity
            current_label = label[2:]  # Remove the "B-" prefix
            current_entity = token.replace('▁', '')  # Handle tokenizer special tokens (e.g., '▁')

        elif label.startswith(
                "I-") and current_label:  # and current_label == label[2:]:  # Continuation of the current entity
            current_entity += token.replace('▁', ' ')  # Append the current token to the entity

        else:
            # If there is no continuation, finalize the current entity
            if current_entity and current_label:
                if current_label not in entities:
                    entities[current_label] = []
                entities[current_label].append(current_entity)

            current_entity = None
            current_label = None

    if current_entity and current_label:
        if current_label not in entities:
            entities[current_label] = []
        entities[current_label].append(current_entity)

    if partial:
        entities = {key: value for key, value in entities.items() if key in RELEVANT_CATEGORIES}

    for key in entities:
        entities[key] = list(set(entities[key]))

    entities['text'] = text
    if train:
        entities['id'] = id
    age = extract_age(entities.get('AGE', []))
    entities['AGE'] = age if age is not None else None
    sex = map_sex(entities.get('SEX', []))
    entities['SEX'] = sex
    return entities


def extract_age(age_list):
    """
    Convert the first age entry in the list to an integer or specific value for known terms.
    Checking for special cases (infant, newborn,etc.), then find the first number in the age string by using regex.
    """
    if age_list:
        age_string = age_list[0].lower()
        if any(term in age_string for term in ['infant', 'newborn', 'neonate', 'baby']):
            return 2
        match = re.search(r'\d+', age_string)
        if match:
            return int(match.group(0))
    return None


def map_sex(sex_list):
    """
    Convert the first sex entry in the list to a boolean if it contains 'female' or 'male'.
    If it's unclear or contains terms like 'infant' or 'neonate', return None.
    """
    if sex_list:
        for sex_value in sex_list:
            for term, is_female in PREDEFINED_SEX_MAPPING.items():  # predefined_sex_mapping.items():
                if term in sex_value and is_female is not None:
                    return is_female
    return None


def start_extraction(data=None, save_to_path=False, extract_from_source=False, dir_path='../data/',
                     doc_source='medqa/test'):
    if extract_from_source or not data:
        target_path = dir_path + doc_source + 'json'
        with open(target_path, 'r') as f:
            data = json.load(f)
    elif data is not None:
        print("Entity extraction pipeline initiated...")

    data_entities = []
    for ID, query in enumerate(data):
        ents = extract_entities(text=query, id=ID)  # !!!!!!!!!! entry['generated_explanation']
        data_entities.append(ents)
    if save_to_path:
        with open('../data/test/test_entities.json', 'w') as f:
            json.dump(data_entities, f, indent=4, default=lambda x: float(x), separators=(',', ': '))
    return data_entities if len(data_entities) > 1 else data_entities[0]
