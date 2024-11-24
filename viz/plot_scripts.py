import networkx as nx
import matplotlib.pyplot as plt

color_dict = {"post_rerank": "#D0E8C5", "common_post_rerank": "#FFCCEA", "not_common_pre_rerank": "#D0E8C5",
              "common_pre_rerank_not_post": "#CDC1FF"}


def plot_rerank_graph(G, ranked_docs, k, n, plot_title="Basic Retreival", save_fig=False):
    """
    Plot the graph G and highlight the top n ranked documents with special markers.

    :param save_fig:
    :param plot_title:
    :param method:
    :param k:
    :param G: NetworkX graph created using HITS.
    :param ranked_docs: List of top-ranked documents (sorted by authority scores).
    :param n: Number of documents to highlight.
    """
    top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
    node_colors = ['#704264' if node in top_n_ids else '#DBAFA0' for node in G.nodes()]
    node_sizes = [400 if node in top_n_ids else 400 for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 6))

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="#6B728E",
        font_size=7,
        font_color="#000000",  # Set node labels to white
        ax=ax
    )
    if n != k:
        plt.scatter([], [], c='#DBAFA0', s=100, label=f"Pre Rerank Retreival || k = {k}")
        plt.scatter([], [], c='#704264', s=100, label=f"Post Rerank || n = {n}")
    else:
        plt.scatter([], [], c='#DBAFA0', s=100, label=f"Retreived Documents || k = {k}")

    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='upper left', bbox_to_anchor=(1, 1))
    plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.99)

    ax.set_position([0.1, 0.1, 0.8, 0.8])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    if plot_title != "Basic Retreival":
        plot_title = plot_title.replace("|", " ")
    if save_fig:
        plt.save_fig(f'plots/{plot_title.lower().replace(" ", "_")}.png')
    plt.show()


# def plot_joined_hits_graphs(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False):
#     """
#     Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
#
#     :param k:
#     :param G1: First NetworkX graph (e.g., for filtered retrieval).
#     :param ranked_docs1: List of top-ranked documents for the first graph.
#     :param G2: Second NetworkX graph (e.g., for basic retrieval).
#     :param ranked_docs2: List of top-ranked documents for the second graph.
#     :param n: Number of documents to highlight in both graphs.
#     """
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 10))
#
#     def get_node_styles(G, ranked_docs):
#         top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
#         node_colors = ['#704264' if node in top_n_ids else '#DBAFA0' for node in G.nodes()]
#         node_sizes = [500 if node in top_n_ids else 300 for node in G.nodes()]
#         return node_colors, node_sizes
#
#     node_colors1, node_sizes1 = get_node_styles(G1, ranked_docs1)
#     node_colors2, node_sizes2 = get_node_styles(G2, ranked_docs2)
#     for i, n in enumerate(G1):
#         if n in G2.nodes():
#             new_nc = "#FDAF7B"
#             node_colors1[i] = new_nc
#     for i, n in enumerate(G2):
#         if n in G1.nodes():
#             new_nc = "#FDAF7B"
#             node_colors2[i] = new_nc
#
#     pos1 = nx.spring_layout(G1)
#     nx.draw(
#         G1,
#         pos1,
#         with_labels=True,
#         node_color=node_colors1,
#         node_size=node_sizes1,
#         edge_color="#6B728E",
#         font_size=7,
#         font_color="#322653",
#         ax=axes[0]
#     )
#     axes[0].set_title("Entities Filtering Retrieval & HITS Rerank")
#
#     pos2 = nx.spring_layout(G2)
#     nx.draw(
#         G2,
#         pos2,
#         with_labels=True,
#         node_color=node_colors2,
#         node_size=node_sizes2,
#         edge_color="#6B728E",
#         font_size=7,
#         font_color="#322653",
#         ax=axes[1]
#     )
#     axes[1].set_title("Basic Retrieval & HITS Rerank")
#
#     plt.scatter([], [], c='#FDAF7B', s=200, label=f'Retrived in both methods')
#     plt.scatter([], [], c='#DBAFA0', s=200, label=f'Retrived Documents')
#     plt.scatter([], [], c='#704264', s=200, label=f" HITS Rerank || n = {n}")
#
#     plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='best')
#
#     plt.suptitle("Retrieval Methods Comparison")
#     plt.tight_layout()
#     if save_fig:
#         plt.save_fig(f'plots/{method.lower().replace(" ", "_")}.png')
#     plt.show()


def plot_graphs_comparison_prev(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                                rerank_title="Entities Reranking Retreival"):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    def get_node_styles(G, ranked_docs, shared_nodes):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#704264' if node in top_n_ids else '#DBAFA0' for node in G.nodes()]
        node_sizes = [500 if node in top_n_ids else 300 for node in G.nodes()]
        edge_colors = ['red' if node in shared_nodes else 'none' for node in G.nodes()]
        return node_colors, node_sizes, edge_colors

    shared_nodes = set(G1.nodes()).intersection(set(G2.nodes()))

    node_colors1, node_sizes1, edge_colors1 = get_node_styles(G1, ranked_docs1, shared_nodes)
    node_colors2, node_sizes2, edge_colors2 = get_node_styles(G2, ranked_docs2, shared_nodes)

    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)

    def draw_graph(ax, G, pos, node_colors, node_sizes, edge_colors):
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="#6B728E",
            font_size=7,
            font_color="#322653",
            ax=ax,
        )
        for node, (x, y) in pos.items():
            if node in shared_nodes:
                circ = plt.Circle((x, y), 0.05, edgecolor='red', facecolor='none', linewidth=2, transform=ax.transData)
                ax.add_patch(circ)

    draw_graph(axes[0], G1, pos1, node_colors1, node_sizes1, edge_colors1)
    axes[0].set_title(rerank_title)

    draw_graph(axes[1], G2, pos2, node_colors2, node_sizes2, edge_colors2)
    axes[1].set_title("Basic Retrieval")

    plt.axvline(x=0.2, color='black', linewidth=2)

    plt.scatter([], [], c='#FDAF7B', s=200, label='Retrieved in both methods')
    plt.scatter([], [], c='#DBAFA0', s=200, label='Retrieved Documents')
    plt.scatter([], [], c='#704264', s=200, label=f"HITS Rerank || n = {n}")

    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='best')
    plt.subplots_adjust(wspace=0.5)
    plt.suptitle("Retrieval Methods Comparison")
    plt.tight_layout()
    if save_fig:
        plt.save_fig(f'plots/retrieval_methods_comparison_k{k}.png')
    plt.show()


def plot_graphs_comparison(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                           rerank_title="Entities Reranking Retrieval"):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line.
    Each subplot has its title in bold, with a text box showing the average score of its ranked documents.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    def get_node_styles(G, ranked_docs, shared_nodes):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#CB9DF0' if node in top_n_ids else '#FFF9BF' for node in G.nodes()]
        font_colors = ['white' if node in top_n_ids else '#1A1A1D' for node in G.nodes()]
        node_sizes = [500 if node in top_n_ids else 300 for node in G.nodes()]
        edge_colors = ['#821131' if node in shared_nodes else 'none' for node in G.nodes()]
        return node_colors, font_colors, node_sizes, edge_colors

    shared_nodes = set(G1.nodes()).intersection(set(G2.nodes()))

    node_colors1, font_colors1, node_sizes1, edge_colors1 = get_node_styles(G1, ranked_docs1, shared_nodes)
    node_colors2, font_colors2, node_sizes2, edge_colors2 = get_node_styles(G2, ranked_docs2, shared_nodes)

    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)

    #     pos1 = nx.spring_layout(G1)
    #     nx.draw(
    #         G1,
    #         pos1,
    #         with_labels=True,
    #         node_color=node_colors1,
    #         node_size=node_sizes1,
    #         edge_color="#6B728E",
    #         font_size=7,
    #         font_color="#000000",
    #         ax=axes[0]
    #     )
    #     axes[0].set_title("Entities Filtering Retrieval & Rerank")
    #
    #     pos2 = nx.spring_layout(G2)
    #     nx.draw(
    #         G2,
    #         pos2,
    #         with_labels=True,
    #         node_color=node_colors2,
    #         node_size=node_sizes2,
    #         edge_color="#6B728E",
    #         font_size=7,
    #         font_color="#000000",
    #         ax=axes[1]
    #     )
    # for node, (x, y) in pos.items():
    #     if node in shared_nodes:
    #         circ = plt.Circle((x, y), 0.05, edgecolor='red', facecolor='none', linewidth=2, transform=ax.transData)
    #         ax.add_patch(circ)
    def draw_graph(ax, G, pos, node_colors, font_colors, node_sizes, edge_colors, ranked_docs, title):
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            font_size=7,
            font_color="#000000",
            ax=ax
        )

        for node, (x, y) in pos.items():
            if node in shared_nodes:
                circ = plt.Circle((x, y), 0.05, edgecolor='red', facecolor='none', linewidth=2, transform=ax.transData)
                ax.add_patch(circ)

        avg_score = sum([doc['score'] for doc in ranked_docs[:n]]) / n
        ax.text(0.5, 0.05, f"Avg Score: {avg_score:.2f}", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7))

        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.set_position([0.1, 0.1, 0.8, 0.8])

        ax.set_xticks([])
        ax.set_yticks([])

    draw_graph(axes[0], G1, pos1, node_colors1, font_colors1, node_sizes1, edge_colors1, ranked_docs1, rerank_title)
    draw_graph(axes[1], G2, pos2, node_colors2, font_colors2, node_sizes2, edge_colors2, ranked_docs2,
               "Basic Retrieval")

    plt.scatter([], [], c='#821131', s=100, label='Retrieved in both methods')
    plt.scatter([], [], c='#DBAFA0', s=100, label='Retrieved Documents')
    plt.scatter([], [], c='#CB9DF0', s=100, label=f"After Rerank || n = {n}")

    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='best')

    plt.subplots_adjust(wspace=0.3)
    plt.suptitle("Retrieval Methods Comparison", fontsize=16, fontweight='bold', y=1)

    if save_fig:
        plt.savefig(f'plots/retrieval_methods_comparison_k{k}.png')

    plt.tight_layout()
    plt.show()


def plot_graphs_comparison8(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                            rerank_title="Entities Reranking Retrieval"):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line.
    Each subplot has its title in bold, with a text box showing the average score of its ranked documents.
    """

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Adjust figsize for clarity

    def get_node_styles(G, ranked_docs, shared_nodes):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#CB9DF0' if node in top_n_ids else '#DBAFA0' for node in G.nodes()]
        node_sizes = [500 if node in top_n_ids else 300 for node in G.nodes()]
        return node_colors, node_sizes

    shared_nodes = set(G1.nodes()).intersection(set(G2.nodes()))

    node_colors1, node_sizes1 = get_node_styles(G1, ranked_docs1, shared_nodes)
    node_colors2, node_sizes2 = get_node_styles(G2, ranked_docs2, shared_nodes)

    pos1 = nx.spring_layout(G1, seed=42)  # Fixed layout for reproducibility
    pos2 = nx.spring_layout(G2, seed=42)

    def draw_graph(ax, G, pos, node_colors, node_sizes, ranked_docs, title):
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color="#6B728E",
            font_color="#000000",
            font_size=8,
            ax=ax
        )

        for node, (x, y) in pos.items():
            if node in shared_nodes:
                circ = plt.Circle((x, y), 0.01, edgecolor='red', facecolor='none', linewidth=2, transform=ax.transData)
                ax.add_patch(circ)

        avg_score = sum([doc['score'] for doc in ranked_docs[:n]]) / n
        ax.text(0.5, -0.1, f"Avg Score: {avg_score:.2f}", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7))

        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

    draw_graph(axes[0], G1, pos1, node_colors1, node_sizes1, ranked_docs1, rerank_title)
    draw_graph(axes[1], G2, pos2, node_colors2, node_sizes2, ranked_docs2, "Basic Retrieval")
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color='#821131', lw=2, label='Retrieved in both methods'),
            plt.Line2D([0], [0], color='#CB9DF0', marker='o', markersize=10, linestyle='',
                       label=f"{n} Reranked Documents"),
            plt.Line2D([0], [0], color='#DBAFA0', marker='o', markersize=10, linestyle='', label="Retrieved Documents")
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=10
    )

    plt.subplots_adjust(wspace=0.3, top=0.85)
    plt.suptitle("Retrieval Methods Comparison", fontsize=16, fontweight='bold')

    if save_fig:
        plt.savefig(f'plots/retrieval_methods_comparison_k{k}.png', bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_graphs_comparison9(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                            rerank_title="Entities Reranking Retreival"):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), width_ratios=[0.5, 0.5])

    def get_node_styles(G, ranked_docs, shared_nodes):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#704264' if node in top_n_ids else '#DBAFA0' for node in G.nodes()]
        node_sizes = [400 if node in top_n_ids else 400 for node in G.nodes()]

        return node_colors, node_sizes,

    shared_nodes = set(G1.nodes()).intersection(set(G2.nodes()))

    node_colors1, node_sizes1 = get_node_styles(G1, ranked_docs1, shared_nodes)
    node_colors2, node_sizes2 = get_node_styles(G2, ranked_docs2, shared_nodes)

    pos1 = nx.spring_layout(G1)
    pos2 = nx.spring_layout(G2)

    def draw_graph(ax, G, pos, node_colors, node_sizes):
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            font_size=8,
            edge_color="#6B728E",
            font_color="#000000",
            ax=ax,
        )
        for node, (x, y) in pos.items():
            if node in shared_nodes:
                circ = plt.Circle((x, y), 0.05, edgecolor='#821131', facecolor='none', linewidth=2,
                                  transform=ax.transData)
                ax.add_patch(circ)

    draw_graph(axes[0], G1, pos1, node_colors1, node_sizes1)
    axes[0].set_title(rerank_title)

    draw_graph(axes[1], G2, pos2, node_colors2, node_sizes2)
    axes[1].set_title("Basic Retrieval")

    plt.scatter([], [], c='#821131', s=100, label='Retrieved in both methods')
    plt.scatter([], [], c='#DBAFA0', s=100, label='Retrieved Documents')
    plt.scatter([], [], c='#704264', s=100, label=f"Post Rerank || n = {n}")
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='lower right', bbox_to_anchor=(1.05, 1))
    plt.suptitle("Retrieval Methods Comparison")
    plt.tight_layout()
    if save_fig:
        plt.save_fig(f'plots/retrieval_methods_comparison_k{k}.png')
    plt.show()
