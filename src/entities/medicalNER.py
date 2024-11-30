import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from config import RELEVANT_CATEGORIES

load_dotenv()
ner_model_name = os.getenv("NER_MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
model = AutoModelForTokenClassification.from_pretrained(ner_model_name)


def extract_entities(text, id: int, partial=True, index_mode=True) -> dict:
    """
    1. Tokenize the input
    2. Get the model output
    3. Extract the logits (raw predictions) and convert them into predicted class IDs
    4. Map predicted IDs to actual entity labels using the tokenizer's label mapping,
    5. Get the mapping of class ids to label names
    6. Initialize an empty dictionary to store results and declare variables to keep track of the current entity,
    7. Process the tokens and their labels

    :param text: text to extract entities
    :param id: chunk ID
    :param partial: True if extraction is limited to
        {"AGE", "SEX", "SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER"},
        False for using all the availavble entities.
    :param index_mode: True if input is part of train set, False for test queries
    :return: Dictionary with the extracted entities.
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

    # Process the tokens and their labels
    for token, predicted_id in zip(tokens, predicted_class_ids[0]):
        label = label_ids[predicted_id.item()]
        token = token.lower()  # Convert to lowercase for consistency

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

            # Reset the current entity
            current_entity = None
            current_label = None

    # Ensure to add the last entity if it exists
    if current_entity and current_label:
        if current_label not in entities:
            entities[current_label] = []
        entities[current_label].append(current_entity)

    # If partial is True, filter only relevant categories
    if partial:
        entities = {key: value for key, value in entities.items() if key in RELEVANT_CATEGORIES}

    # Remove duplicates by converting lists to sets and back to lists
    for key in entities:
        entities[key] = list(set(entities[key]))

    entities['text'] = text
    entities['id'] = id

    if index_mode:
        entities['id'] = id

    return entities


def extract_from_json():
    """
    For a given JSON filem process each entry to extract entities, and save to path.
    """

    data_dir = '../data/'

    for source in ['medmcqa/', 'medqa/', 'try/']:
        with open(f'{data_dir}{source}train.json', 'r') as f:
            data = json.load(f)
            break

    data_entities = [extract_entities(entry['generated_explanation'], id) for id, entry in enumerate(data)]
    print('Num recordes annotated with NER:', len(data_entities))

    with open('../data/medmcqa/train_entities.json', 'w') as f:
        json.dump(data_entities, f, indent=4, default=lambda x: float(x), separators=(',', ': '))


if __name__ == "__main__":
    extract_from_json()
    # medqa 1018
    # medmcqa 3330
