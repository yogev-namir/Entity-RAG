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


def rerank(G, retrived_k_docs, n, method='HITS', metric='hubs'):
    if method == 'HITS':
        hits_scores = nx.hits(G, normalized=True)
        if metric == 'hubs':
            ranked_docs = apply_hits(hits_scores=hits_scores, retrieved_docs=retrived_k_docs, n=n, metric='hubs')
        else:
            ranked_docs = apply_hits(hits_scores=hits_scores, retrieved_docs=retrived_k_docs, n=n, metric='auth')
    else:
        ranked_docs = apply_pagerank(G, retrieved_k_docs=retrived_k_docs, n=n)

    return G, ranked_docs


def apply_hits(hits_scores, retrieved_docs, n, metric="hubs"):
    hubs_scores = hits_scores[0]
    authority_scores = hits_scores[1]
    if metric == 'hubs':
        ranked_docs = sorted(
            retrieved_docs,
            key=lambda doc: hubs_scores.get(doc['id'], 0),
            reverse=True
        )
    else:  # metric == 'auth':
        ranked_docs = sorted(
            retrieved_docs,
            key=lambda doc: authority_scores.get(doc['id'], 0),
            reverse=True
        )
    return ranked_docs[:n]


def apply_pagerank(G, retrieved_k_docs, n):
    """
    Apply PageRank to the graph and rank the retrieved documents.

    :param G: NetworkX graph.
    :param retrieved_k_docs: List of retrieved documents, each with an 'id'.
    :param n: Number of top-ranked documents to return.
    :return: The graph with updated PageRank scores and the top `n` ranked documents.
    """

    pr_scores = nx.pagerank(G, alpha=0.95)
    ranked_docs = sorted(retrieved_k_docs, key=lambda doc: pr_scores.get(doc['id'], 0), reverse=True)
    return ranked_docs[:n]
