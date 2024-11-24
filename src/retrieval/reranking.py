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
    Sort the results based on edge values and weights in descentding order.
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
        reverse=True
    )
    return sorted_results


def structure_response(sorted_results):
    """
    Structure the response based on sorted results.
    Assume the answer (e.g., a letter or symbol) is accessible in the metadata, replace 'answer' with the
    right answer option.
    :param sorted_results: Sorted list of retrieved documents
    :return: Structured explanation and chosen answer
    """
    if not sorted_results:
        return "No relevant results found.", None

    top_result = sorted_results[-1]  # [0] ?
    explanation = (
        f"Chosen result explanation: Based on context '{top_result['metadata'].get('text', '')}' "
        f"with edge value {top_result['metadata'].get('edge_value', 'N/A')} "
        f"and weight {top_result['metadata'].get('weight', 'N/A')}."
    )

    chosen_answer = top_result['metadata'].get('answer', 'N/A')

    return explanation, chosen_answer




def hits_rerank(original_query: str, retrieved_k_docs: list, n: int):
    """
    Rerank the top k retrieved documents using the HITS algorithm.
    :param original_query: The original query string.
    :param retrieved_k_docs: List of dictionaries containing 'id' and 'metadata' for retrieved documents.
                             Each document metadata should contain a list of references or related documents.
    :return: A list of the 10 most relevant documents (sorted by authority scores).
    """
    G = nx.DiGraph()

    # Step 2: Add nodes for each document
    for doc in retrieved_k_docs:
        doc_id = doc['id']
        G.add_node(doc_id, metadata=doc.get('metadata', {}))

    # Step 3: Add edges based on document relationships (e.g., references or links)
    for doc in retrieved_k_docs:
        doc_id = doc['id']
        related_docs = retrieve_from_index(doc['metadata']['text'])
        for r_doc in related_docs:
            related_doc_id = r_doc['id']
            if related_doc_id in G.nodes:  # Only link to documents in the retrieved set
                G.add_edge(doc_id, related_doc_id)

    hits_scores = nx.hits(G, normalized=True)
    authority_scores = hits_scores[1]

    # Step 5: Sort documents by authority scores
    ranked_docs = sorted(
        retrieved_k_docs,
        key=lambda doc: authority_scores.get(doc['id'], 0),
        reverse=True
    )

    return ranked_docs[:n]


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
