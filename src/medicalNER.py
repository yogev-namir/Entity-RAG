import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# Load the tokenizer and model for token classification
tokenizer = AutoTokenizer.from_pretrained("Clinical-AI-Apollo/Medical-NER")
model = AutoModelForTokenClassification.from_pretrained("Clinical-AI-Apollo/Medical-NER")

# Define relevant categories for 'partial' mode
RELEVANT_CATEGORIES = {"AGE", "SEX", "SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER"}


def extract_entities(text, id, partial=True):
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")

    # Get the model output
    outputs = model(**inputs)

    # Extract the logits (raw predictions)
    logits = outputs.logits

    # Convert the logits into predicted class IDs
    predicted_class_ids = torch.argmax(logits, dim=2)

    # Map predicted IDs to actual entity labels using the tokenizer's label mapping
    label_ids = model.config.id2label  # Get the mapping of class ids to label names
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Initialize an empty dictionary to store results
    entities = {}

    # Variables to keep track of the current entity
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

        elif label.startswith("I-") and current_label:#and current_label == label[2:]:  # Continuation of the current entity
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
    
    return entities    



def main():
    # Define a sample medical text
    #entities = extract_entities( "Feed forward mechanism.. Salivation on smelling food. Here is an additional explanation: Feed forward mechanism Feedback mechanism Controller anticipates changes & takes a desired action. No time lag present Examples: - Cephalic phase of gastric acid secretion Increase ventilatory drive in exercise. Change occur in controlled variable & that change is feedback to controller & then the controller takes action Time lag is present. Type: Negative feed back - Kidney body fluid mechanism - Temperature regulation - Baroreceptor mechanism Positive feed back - Circulatory shock - Oxytocin in paurition - Platelet plug / clot formation - LH surge leading to ovulation - Bladder filling to micturition", 0)
    #print(entities)
    #stop

    data_dir = 'data/'

    for source in ['medmcqa/','medqa/', 'try/' ]:
        with open(f'{data_dir}{source}train.json', 'r') as f:
            data = json.load(f)
            break

    # Process each JSON entry to extract entities
    data_entities = [extract_entities(entry['generated_explanation'], id) for id, entry in enumerate(data)]
    print('Num recordes annotated with NER:', len(data_entities))

    # Save the processed data with entities to a new file
    with open('data/medmcqa/train_entities.json', 'w') as f:
        json.dump(data_entities, f, indent=4, default=lambda x: float(x), separators=(',', ': '))

if __name__ == "__main__":
    main()
    # medqa 1018
    # medmcqa 3330