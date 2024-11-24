import pandas as pd


def prec_k(query, retrived_doc, k):
    pass


def accuracy_k(query, retrived_doc, k):
    pass


def zero_results_rate():
    pass


def calc_LSI():
    """
    Latent semantic indexing (LSI) is an indexing and retrieval method that uses a mathematical technique called
    singular value decomposition (SVD) to identify patterns in the relationships between the terms and concepts
    contained in an unstructured collection of text.
    :return:
    """


def calc_relevance():
    pass


def entities_df(query=[], docs=[]):
    """
    Create entities df for retrived queries -
    columns - [chunk_id, text, symptoms,desease,biological_stracture, medications]
    :param query:
    :param docs:
    :return:
    """
    file_path = "../src/data/medmcqa/train_entities.json"
    print()
    df = pd.read_json(file_path)
    df.to_csv("../src/data/medmcqa/dfs/short_corpus_entities.csv", index=False)


def query_entities_graph(doc):
    edge_key = doc['id']
    edge_weight = doc['score']

    edges = pd.DataFrame(
        {
            "source": [0, 1, 2, 0],
            "target": [2, 2, 3, 2],
            "my_edge_key": ["A", "B", "C", "D"],
            "weight": [3, 4, 5, 6],
            "color": ["red", "blue", "blue", "blue"],
        }
    )
    G = nx.from_pandas_edgelist(
        edges,
        edge_key="my_edge_key",
        edge_attr=["weight", "color"],
        create_using=nx.MultiGraph(),
    )


entities_df()
