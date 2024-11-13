import collections
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from src.indexing.vdb_indexing import *
from src.entities.medicalNER import *

load_dotenv()
host_url = os.getenv('HOST_URL')
pc_api_key = os.getenv('PINECONE_API_KEY')
model_name = os.getenv('EMBEDDING_MODEL_NAME')
name_space = os.getenv('NAME_SPACE')
index_name = os.getenv('INDEX_NAME')
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


def sort_by_edge_values_and_weights(results, edge_key='edge_value', weight_key='weight'):
    """
    Sort the results based on edge values and weights.
    :param results: List of retrieved documents with metadata
    :param edge_key:  key for edge values
    :param weight_key: Metadata key for weights
    :return: Sorted list of results
    """
    sorted_results = sorted(
        results,
        key=lambda x: (
            x['metadata'].get(edge_key, 0),
            x['metadata'].get(weight_key, 0)
        ),
        reverse=True  # Descending order (if higher values are better)
    )
    return sorted_results


def structure_response(sorted_results):
    """
    Structure the response based on sorted results.
    :param sorted_results: Sorted list of retrieved documents
    :return: Structured explanation and chosen answer
    """
    if not sorted_results:
        return "No relevant results found.", None

    # Select the top result for explanation
    top_result = sorted_results[-1]  # [0]
    explanation = (
        f"Chosen result explanation: Based on context '{top_result['metadata'].get('text', '')}' "
        f"with edge value {top_result['metadata'].get('edge_value', 'N/A')} "
        f"and weight {top_result['metadata'].get('weight', 'N/A')}."
    )

    # Assume the answer (e.g., a letter or symbol) is accessible in the metadata
    chosen_answer = top_result['metadata'].get('answer', 'N/A')  # Replace 'answer' with the actual key if different

    return explanation, chosen_answer


if __name__ == "__main__":
    query_text = """A 23-year-old woman comes to the physician because of a 2-month history of episodic headaches 
    associated with dizziness, nausea, and vomiting. Over-the-counter pain medications have failed to reduce her 
    symptoms. " "An MRI of the brain shows isolated dilation of the left lateral ventricle. " "This dilatation is 
    most likely caused by blockade of which of the following structures? {"opa": "Arachnoid villi", 
    "opb": "Interventricular foramen", "opc": "Median aperture", "opd": "Lateral apertures"}"""
    top_k = 5

    retrieved_docs = retrieve_from_index(query_text, top_k=top_k)
    sorted_results = sort_by_edge_values_and_weights(retrieved_docs)
    explanation, chosen_answer = structure_response(sorted_results)

    for idx, doc in enumerate(retrieved_docs):
        print(f"Document {idx + 1}:")
        print(f"ID: {doc['id']}")
        print(f"Score: {doc['score']}")
        print(f"Metadata: {doc['metadata']}\n")
    print(explanation)
    print(f"Chosen answer: {chosen_answer}")
