def prepare_metadata(entry):
    """
    Prepearing metadata for the given entry by removing None or empty values in the metadata
    :param entry: entry to process
    :return: metadata for entry
    """
    metadata = {}

    if entry.get('AGE') is not None:
        metadata['AGE'] = entry['AGE']

    if entry.get('SEX') is not None:
        metadata['SEX'] = entry['SEX']

    if entry.get('SIGN_SYMPTOM'):
        metadata['SIGN_SYMPTOM'] = entry['SIGN_SYMPTOM']

    if entry.get('MEDICATION'):
        metadata['MEDICATION'] = entry['MEDICATION']

    if entry.get('BIOLOGICAL_STRUCTURE'):
        metadata['BIOLOGICAL_STRUCTURE'] = entry['BIOLOGICAL_STRUCTURE']

    if entry.get('DISEASE_DISORDER'):
        metadata['DISEASE_DISORDER'] = entry['DISEASE_DISORDER']

    return metadata


def construct_metadata(entry, embedding):
    vec = {
        "id": str(entry['id']),
        "values": embedding['values'],
        "metadata": {
            "text": entry['text'],
            **metadata
        }
    }
    return vec


def add_to_index(entries, embeddings):
    """
    Adding cleaned metadata to index, then updating and inserting.
    :param entries:
    :param embeddings:
    :return: vectors
    """
    vectors = []
    for entry, embedding in zip(entries, embeddings):
        vectors.append(construct_metadata(entry, embedding))
    return vectors
