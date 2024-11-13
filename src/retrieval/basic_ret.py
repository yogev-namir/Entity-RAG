import collections
import os
from src.indexing.vectorDB_indexing import *
from sentence_transformers import SentenceTransformer
from src.entities.entities_extraction import *
from config import TEST_QUERIES

load_dotenv()
host_url = os.getenv('HOST_URL')  # "https://entity-metadata-index-1024-mcmd0qy.svc.aped-4627-b74a.pinecone.io"
pc_api_key = os.getenv('PINECONE_API_KEY')  # 'bb68c35d-a2f2-47a7-9d21-78f0e3b0ab68'
index_name = os.getenv('INDEX_NAME')  # "entity-metadata-index-1024"  # !!!!!
model_name = os.getenv('EMBEDDING_MODEL_NAME')
name_space = os.getenv('NAME_SPACE')
pc = init_client(pc_api_key)
model = SentenceTransformer(model_name)
index = pc.Index(name=index_name, host=host_url)


def retrieve_from_index(query, top_k=5):
    """
    Retrieve top-k documents based on the query embedding.
    :param query: Text query
    :param top_k: Number of top results to retrieve
    :return: List of retrieved documents
    """

    model = SentenceTransformer('intfloat/multilingual-e5-large')
    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        namespace=name_space
    )

    retrieved_docs = []
    if 'matches' in results:
        for match in results['matches']:
            doc = {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            retrieved_docs.append(doc)

    return retrieved_docs


def metadate_filter(query, query_entities, k=10):
    query_embedding = model.encode(query).tolist()
    query_filters = collections.defaultdict()
    filter_by = ["SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER"]
    # filter_by = ["MEDICATION", "DISEASE_DISORDER"]
    for key, values in query_entities.items():
        if values is not None and (key in filter_by):
            if type(values) is int:
                values = [values]
            elif type(values) is bool:
                values = ['feminin'] if values else ['masculin']
            query_filters[key] = {"$in": values}
    results = index.query(
        namespace=name_space,
        vector=query_embedding,
        filter=query_filters,
        top_k=k,
        include_values=False,
        include_metadata=True
    )

    retrieved_docs = []
    if 'matches' in results:
        for match in results['matches']:
            doc = {
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata']
            }
            retrieved_docs.append(doc)

    return retrieved_docs


for test in TEST_QUERIES:
    query = test['generated_explanation']
    query_entities = start_extraction(data=[query], index_mode=False)
    print(
        "=============================================================================================================================")
    print(f"Query: {query}")
    print(query_entities)
    print(
        "=============================================================================================================================")
    top_relevant_docs = metadate_filter(query, query_entities)
    for i, doc in enumerate(top_relevant_docs):
        print(f"Document {i} | Score = {doc['score']}")
        metadata = doc['metadata']
        if 'SEX' in metadata.keys():
            sex = 'women' if metadata['SEX'] else 'men'
            print(f"Sex: {sex}")
        if 'AGE' in metadata.keys():
            print(f"Age: {metadata['AGE']}")
        if 'SIGN_SYMPTOM' in metadata.keys():
            print(f"Sign Symptoms: {metadata['SIGN_SYMPTOM']}")
        if 'DISEASE_DISORDER' in metadata.keys():
            print(f"DISEASE / DISORDER: {metadata['DISEASE_DISORDER']}")
        if 'MEDICATION' in metadata.keys():
            print(f"Medications: {metadata['MEDICATION']}")
        if 'BIOLOGICAL_STRUCTURE' in metadata.keys():
            print(f"Biological Structure: {metadata['BIOLOGICAL_STRUCTURE']}")
        print(f"Text: {metadata['text']}")
        print("==============================================================================================")
    print(f"\n\n\n\n\n\n")
