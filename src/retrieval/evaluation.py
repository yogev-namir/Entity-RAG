import pandas as pd


def start_experiments(test_samples: pd.DataFrame, methods=None, metrics=None):
    if methods is None:
        methods = ["HITS"]
        if metrics is None:
            metrics = ["hubs"]
    for method in methods:
        if method == 'PageRank':
            methods_rspn['pr_rspn'] = []
        elif 'auth' not in metrics:
            methods_rspn[method]['hubs_rspn'] = {}
        else:
            methods_rspn[method]['auth_rspn'] = {}
    methods_rspn['benchmark'] = []
    hubs_rspn = []
    auth_rspn = []
    pr_rspn = []
    methods = ["HITS"]  # ["PageRank", "HITS"]
    metrics = ["hubs"]  # ["hubs", "auth"]
    for index, test in test_samples.iterrows():
        print("======================================================================")
        query = test['query']
        print(query)
        print("======================================================================")
        k, n = 25, 5
        G_basic, top_n_docs, shared_entities_basic = basic_approach(query, k=10, n=5)
        G_filter, filter_retrieval_docs, shared_entities_filter = filter_approach(query, k=k, n=n)
        for method in methods:
            try:
                if method == "PageRank":
                    G_filter, reranked_docs_pr = rerank(G_filter, filter_retrieval_docs, n=n, method=method)
                    compare_graphs(G_filter, reranked_docs_pr, G_basic, top_n_docs, k, n, method=method)
                    pr_rspn.append(generate_response(query, reranked_docs_pr))
                elif method == "HITS":
                    for metric in metrics:
                        if metric == 'hubs':
                            G_filter, reranked_docs_hubs = rerank(G_filter, filter_retrieval_docs, n=n, method=method,
                                                                  metric=metric)
                            compare_graphs(G_filter, reranked_docs_hubs, G_basic, top_n_docs, k, n, method=method,
                                           metric=metric)
                            hubs_answer = generate_response(query, reranked_docs_hubs)
                            hubs_rspn.append(hubs_answer)
                        else:
                            G_filter, reranked_docs_auth = rerank(G_filter, filter_retrieval_docs, n=n, method=method,
                                                                  metric=metric)
                            compare_graphs(G_filter, reranked_docs_auth, G_basic, top_n_docs, k, n, method=method, metric=metric)
                            auth_rspn.append(generate_response(query, reranked_docs_auth))
            except Exception as e:
                print(f"Error in reranking or comparing graphs (method={method}): {e}")
                continue
        benchmark_rspn.append(benchmark_response(query, top_n_docs))
