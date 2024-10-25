import json
import re
import os

# Predefined mapping for common SEX terms, enriched
predefined_sex_mapping = {
    "female": True,
    "woman": True,
    "girl": True,
    "lady": True,
    "mother": True,
    "bride": True,
    
    "male": False,
    "man": False,
    "boy": False,
    "gentleman": False,
    "males": False,
    "groom": False,

    # Terms that should map to None (as they are not gender-specific)
    "neonate": None,
    "infant": None,
    "baby": None,
    "newborn": None,
    "child": None,
    "teenager": None,
    "youth": None
}

def extract_age(age_list):
    """
    Convert the first age entry in the list to an integer or specific value for known terms.
    """
    if age_list:
        # Special cases for 'infant', 'newborn', etc., to translate to age 2
        age_string = age_list[0].lower()
        if any(term in age_string for term in ['infant', 'newborn', 'neonate', 'baby']):
            return 2
        # Use regular expression to find the first number in the age string
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
            for term, is_female in predefined_sex_mapping.items():
                if term in sex_value and is_female is not None:
                    return is_female
    return None

def process_data(input_file, output_file):
    """
    Process the input file to convert AGE to integer and map SEX to boolean or None.
    Save the processed data to an output file.
    """
    # Load the original data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Process each entry
    processed_data = []
    for entry in data:
        processed_entry = entry.copy()

        # Process AGE
        age = extract_age(entry.get('AGE', []))
        if age is not None:
            processed_entry['AGE'] = age
        else:
            processed_entry['AGE'] = None    

        # Process SEX
        sex = map_sex(entry.get('SEX', []))
        processed_entry['SEX'] = sex

        # Append processed entry
        processed_data.append(processed_entry)

    # Save the processed data to the output file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    output_dir = 'data/medqa/'

    # Process the medqa dataset
    process_data(
        input_file=f'{output_dir}train_entities.json',
        output_file=f'{output_dir}train_entities_demographics.json'
    )

    output_dir = 'data/medmcqa/'

    # Process the medmcqa dataset
    process_data(
        input_file=f'{output_dir}train_entities.json',
        output_file=f'{output_dir}train_entities_demographics.json'
    )