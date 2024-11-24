import warnings

import cohere
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import TEST_QUERIES, APP_QUERIES
from src.entities.entities_extraction import *
from src.indexing.vectorDB_indexing import *
from viz.plot_scripts import *

warnings.filterwarnings("ignore")

load_dotenv()
host_url = os.getenv('HOST_URL')
pc_api_key = os.getenv('PINECONE_API_KEY')
model_name = os.getenv('EMBEDDING_MODEL_NAME')
rerank_model_name = os.getenv('RERANK_MODEL_NAME')
name_space = os.getenv('NAME_SPACE')
index_name = os.getenv('INDEX_NAME')
cohere_api_key = os.getenv('COHERE_API_KEY')
pc = init_client(pc_api_key)
model = SentenceTransformer(model_name)
index = pc.Index(name=index_name, host=host_url)
co = cohere.Client(api_key=cohere_api_key)


def init_client():
    """
    Initialize Pinecone client
    :return: pc object
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc


def basic_ret(query, k=10):
    """
    Retrieve top-k documents based on the query embedding.
    :param query: Text query
    :param top_k: Number of top results to retrieve
    :return: List of retrieved documents
    """

    query_embedding = model.encode(query).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=k,
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
                'text': match['metadata']['text']
            }
            retrieved_docs.append(doc)

    return retrieved_docs


def metadata_filter_ret(query, query_entities, k=10,
                        filter_by=("SIGN_SYMPTOM", "MEDICATION", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER")):
    """
    Filters retrieved documents based on query entities and their metadata.

    :param query: Input query string.
    :param query_entities: Dictionary of entities extracted from the query.
    :param k: Number of top documents to retrieve for each filter.
    :param filter_by: List of metadata keys to filter by.
    :return: List of retrieved documents with metadata and scores.
    """
    query_embedding = model.encode(query).tolist()
    current_filter = None
    retrieved_docs = []
    retrieved_idx = set()
    filter_by = set(query_entities.keys()).intersection(set(filter_by))
    for key, values in query_entities.items():
        if key in filter_by and values is not None:
            if isinstance(values, int):
                values = [values]
            elif isinstance(values, bool):
                values = ['feminin'] if values else ['masculin']
            current_filter = {key: {"$in": values}}

        results = index.query(
            namespace=name_space,
            vector=query_embedding,
            filter=current_filter,
            top_k=k,
            include_values=False,
            include_metadata=True
        )

        if 'matches' in results:
            for match in results['matches']:
                if match['id'] not in retrieved_idx:
                    doc = {
                        'id': match['id'],
                        'score': match['score'],
                        'metadata': match['metadata'],
                        'embedding': match.get('values')
                    }
                    retrieved_docs.append(doc)
                    retrieved_idx.add(match['id'])

    complete_to_k = k - len(retrieved_idx)
    if complete_to_k > 0:
        results = index.query(
            namespace=name_space,
            vector=query_embedding,
            top_k=complete_to_k,
            include_values=False,
            include_metadata=True
        )

        if 'matches' in results:
            for match in results['matches']:
                if match['id'] not in retrieved_idx:
                    doc = {
                        'id': match['id'],
                        'score': match['score'],
                        'metadata': match['metadata'],
                        'embedding': match.get('values')
                    }
                    retrieved_docs.append(doc)
                    retrieved_idx.add(match['id'])

    return retrieved_docs


#
# def metadata_filter_ret(query, query_entities, k=10,
#                         filter_by=["SIGN_SYMPTOM", "BIOLOGICAL_STRUCTURE", "DISEASE_DISORDER"]):
#     query_embedding = model.encode(query).tolist()
#     query_filters = collections.defaultdict()
#     retrieved_docs, retrived_idx = [], []
#     for key, values in query_entities.items():
#         if values is not None and (key in filter_by):
#             if type(values) is int:
#                 values = [values]
#             elif type(values) is bool:
#                 values = ['feminin'] if values else ['masculin']
#             query_filters[key] = {"$in": values}
#         results = index.query(
#             namespace=name_space,
#             vector=query_embedding,
#             filter=query_filters,
#             top_k=k,
#             include_values=False,
#             include_metadata=True
#         )
#         if 'matches' in results:
#             for match in results['matches']:
#                 if match['id'] not in retrived_idx:
#                     doc = {
#                         'id': match['id'],
#                         'score': match['score'],
#                         'metadata': match['metadata'],
#                         'embedding': match['values']
#                     }
#                     retrieved_docs.append(doc)
#                     retrived_idx.append(match['id'])
#     return retrieved_docs


def compute_graph(original_query: str, retrieved_k_docs: list, k: int, n: int, basic=False):
    """
    Rerank the top k retrieved documents using the HITS algorithm.
    :param k:
    :param n:
    :param basic:
    :param original_query: The original query string.
    :param retrieved_k_docs: List of dictionaries containing 'id' and 'metadata' for retrieved documents.
                             Each document metadata should contain a list of references or related documents.
    :return: A list of the 10 most relevant documents (sorted by authority scores).
    """
    print(f"Computing Graph... (basic retrieval = {basic})")
    G = nx.DiGraph()
    for doc in retrieved_k_docs:
        doc_id = doc['id']
        G.add_node(doc_id, metadata=doc.get('metadata', {}), weight=doc['score'])

    for doc in retrieved_k_docs:
        doc_id = doc['id']
        doc_text = doc['text'] if basic else doc['metadata']['text']
        if basic:
            related_docs = basic_ret(query=doc_text, k=k)
        else:
            doc_entities = start_extraction(data=[doc_text], index_mode=False)
            related_docs = basic_ret(query=doc_text,k=k)  # metadata_filter_ret(query=doc_text, query_entities=doc_entities, k=k)
        for r_doc in related_docs:
            related_doc_id = r_doc['id']
            if related_doc_id != doc_id and related_doc_id in G.nodes:
                G.add_edge(doc_id, related_doc_id, weight=r_doc['score'])

    if basic:
        ranked_docs = sorted(retrieved_k_docs, key=lambda doc: doc['score'], reverse=True)
        return G, ranked_docs[:n]
    return G, retrieved_k_docs


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


def compare_graphs(G_filter, reranked_docs, G_basic, top_n_docs, k=50, n=10, method='HITS', metric='hubs'):
    plot_title = f"Entities Filtering Retreival\n | {method} Reranking |" if method == 'PageRank' else \
        f"Entities Filtering Retreival\n| {method} Reranking | Reorder by {metric.capitalize()} |"
    plot_rerank_graph(G_filter, reranked_docs, k=k, n=n, plot_title=plot_title)
    plot_graphs_comparison9(G_filter, reranked_docs, G_basic, top_n_docs, k=k, n=n, rerank_title=plot_title)


def basic_approach(query, k=10, n=10):
    basic_retrieval_docs = basic_ret(query, k=k)
    G_basic, top_n_docs = compute_graph(original_query=query, retrieved_k_docs=basic_retrieval_docs, k=k, n=n,
                                        basic=True)
    plot_rerank_graph(G_basic, top_n_docs, k=k, n=n, plot_title="Basic Retreival")
    return G_basic, top_n_docs


def filter_approach(query, k=10, n=10):
    query_entities = start_extraction(data=[query], index_mode=False)
    filter_retrieval_docs = metadata_filter_ret(query, query_entities, k=k)
    G_filter, _ = compute_graph(original_query=query, retrieved_k_docs=filter_retrieval_docs, k=k, n=n,
                                basic=False)
    return G_filter, filter_retrieval_docs


# tests = pd.read_csv('mini_test.csv')
# for test in tests.sample(5):
#     query = test['mod_question']
#     k, n = 25, 5
#     G_basic, top_n_docs = basic_approach(query, k=10, n=10)
#     G_filter, filter_retrieval_docs = filter_approach(query, k=k, n=n)
#
#     for method in ["PageRank", "HITS"]:
#         if method == "PageRank":
#             G_filter, reranked_docs = rerank(G_filter, filter_retrieval_docs, n=n, method=method)
#             compare_graphs(G_filter, reranked_docs, G_basic, top_n_docs, k, n, method=method)
#         elif method == "HITS":
#             for metric in ["hubs", "auth"]:
#                 G_filter, reranked_docs = rerank(G_filter, filter_retrieval_docs, n=n, method=method, metric=metric)
#                 compare_graphs(G_filter, reranked_docs, G_basic, top_n_docs, k, n, method=method, metric=metric)
#     reranked_docs


app_queries = False
if app_queries:
    for query in APP_QUERIES:
        print(f"\n\n{query}\n\n")
        k, n = 25, 5
        G_basic, top_n_docs = basic_approach(query, k=10, n=10)
        G_filter, filter_retrieval_docs = filter_approach(query, k=k, n=n)

        for method in ["PageRank", "HITS"]:
            if method == "PageRank":
                G_filter, reranked_docs = rerank(G_filter, filter_retrieval_docs, n=n, method=method)
                compare_graphs(G_filter, reranked_docs, G_basic, top_n_docs, k, n, method=method)
            elif method == "HITS":
                for metric in ["hubs", "auth"]:
                    G_filter, reranked_docs = rerank(G_filter, filter_retrieval_docs, n=n, method=method, metric=metric)
                    compare_graphs(G_filter, reranked_docs, G_basic, top_n_docs, k, n, method=method, metric=metric)

prev = False
if prev:
    filter_ret = True
    reranking_ret = False

    if filter_ret:
        for query in APP_QUERIES:
            metadatas = []
            query = query
            query_entities = start_extraction(data=[query], index_mode=False)
            print(
                "=============================================================================================================================")
            print(f"Query: {query}")
            print(query_entities)
            print(
                "=============================================================================================================================")
            top_relevant_docs = metadata_filter_ret(query, query_entities, k=500)
            for i, doc in enumerate(top_relevant_docs):
                print(f"Document {i} | ID = {doc['id']} | Score = {doc['score']}")
                print(doc['embedding'])
                metadata = doc['metadata']
                metadatas.append(metadata)
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

    elif reranking_ret:
        for test in TEST_QUERIES:
            query = test['question']
            compare_docs(query, 50, 25)
