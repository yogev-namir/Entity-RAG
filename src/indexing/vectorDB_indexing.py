import json
import itertools
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv
import os


def init_client(api_key=None):
    """
    Initialize Pinecone client
    :param api_key:
    :return:
    """
    PINECONE_API_KEY = api_key
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def create_index(pc: Pinecone, index_name: str, dimension: int, metric="cosine"):
    """
    Create a Pinecone index, if doesn't exist
    :param pc: pinecone object
    :param index_name:
    :param dimension:
    :param metric:
    :return:
    """
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


def prepare_metadata(entry):
    """
    Prepearing metadata for the given entry by removing None or empty values in the metadata
    :param entry: entry to process
    :return: metadata for entry
    """
    #
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


def add_to_index(api_key, entries, embeddings, index_name='entity-metadata-index'):
    """
    Adding cleaned metadata to index, then updating and inserting.
    :param api_key:
    :param entries:
    :param embeddings:
    :param index_name:
    :return:
    """
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    vectors = []
    for entry, embedding in zip(entries, embeddings):
        metadata = prepare_metadata(entry)
        vectors.append({
            "id": str(entry['id']),
            "values": embedding['values'],
            "metadata": {
                "text": entry['text'],
                **metadata
            }
        })
    index.upsert(vectors=vectors, namespace="ns0")


def chunks(iterable, batch_size=200):
    """
    Helper function to break an iterable object into chunks
    :param iterable: iterable object
    :param batch_size: size of batch
    :return:
    """
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def main():
    PINECONE_API_KEY = 'bb68c35d-a2f2-47a7-9d21-78f0e3b0ab68'
    index_name = "entity-metadata-index-1024"  # !!!!!
    dimension = 1024

    # load_dotenv()
    # pc_api_key = os.getenv('PINECONE_API_KEY')
    # index_name = os.getenv('INDEX_NAME')
    pc = init_client(PINECONE_API_KEY)

    # Uncomment to create the index (only needs to be done once)
    # create_index(pc, index_name, dimension)

    # Load the data with extracted entities
    data_file_medqa = '../data/medqa/train_entities_demographics.json'
    data_file_medmcqa = '../data/medmcqa/train_entities_demographics.json'

    with open(data_file_medqa, 'r') as f:
        data_medqa = json.load(f)

    with open(data_file_medmcqa, 'r') as f:
        data_medmcqa = json.load(f)

    # Combine both datasets
    data_full = data_medqa + data_medmcqa
    len_data_full = len(data_full)

    for i, j in zip(range(30, len_data_full, 30), range(60, len_data_full, 30)):
        data = data_full[i:j]  # For testing purposes, only use a subset of the data

        # Example: generating embeddings using a model
        # You would need an embedding model to generate embeddings for the texts
        # Replace this with actual embedding generation using your preferred model
        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[d['text'] for d in data],
            parameters={"input_type": "passage", "truncate": "END"},
        )
        print('embedding dim:', len(embeddings[0]['values']))

        # Upsert in batches
        for data_chunk, embeddings_chunk in zip(chunks(data, 1), chunks(embeddings, 1)):
            add_to_index(PINECONE_API_KEY, data_chunk, embeddings_chunk, index_name=index_name)
            # Sleep for 1 second
            time.sleep(10)

        print(f"Data[{i}:{j}] added to the Pinecone index!")


if __name__ == "__main__":
    main()
