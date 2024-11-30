import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    node_colors = ['#D0E8C5' if node in top_n_ids else '#DDF2FD' for node in G.nodes()]

    node_sizes = [500 if node in top_n_ids else 400 for node in G.nodes()]
    fig, ax = plt.subplots(figsize=(10, 6))

    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="#D8D9DA",
        font_size=7,
        font_color="#000000",
        ax=ax
    )

    if n != k:
        plt.scatter([], [], c='#DDF2FD', s=100, label=f"Pre Rerank Retreival || k = {k}")
        plt.scatter([], [], c='#D0E8C5', s=100, label=f"Post Rerank || n = {n}")
    else:
        plt.scatter([], [], c='#DDF2FD', s=100, label=f"Retreived Documents || k = {k}")

    ax.set_position([0.1, 0.1, 0.9, 0.9])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='upper left', bbox_to_anchor=(1, 1))
    plt.suptitle(plot_title, fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    if plot_title != "Basic Retreival":
        plot_title = plot_title.replace("|", " ")
    if save_fig:
        plt.save_fig(f'plots/{plot_title.lower().replace(" ", "_")}.png')
    plt.show()



def plot_graphs_comparison_prev(G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                                rerank_title="Entities Reranking Retreival"):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    def get_node_styles(G, ranked_docs, shared_nodes):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#D0E8C5' if node in top_n_ids else '#DDF2FD' for node in G.nodes()]
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
            edge_color="#D8D9DA",
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
    plt.scatter([], [], c='#DDF2FD', s=200, label='Retrieved Documents')
    plt.scatter([], [], c='#D0E8C5', s=200, label=f"HITS Rerank || n = {n}")

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
    plt.scatter([], [], c='#DDF2FD', s=100, label='Retrieved Documents')
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
        node_colors = ['#CB9DF0' if node in top_n_ids else '#DDF2FD' for node in G.nodes()]
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
            edge_color="#D8D9DA",
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
            plt.Line2D([0], [0], color='#DDF2FD', marker='o', markersize=10, linestyle='', label="Retrieved Documents")
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
        node_colors = ['#D0E8C5' if node in top_n_ids else '#DDF2FD' for node in G.nodes()]
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
            edge_color="#D8D9DA",
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
    plt.scatter([], [], c='#DDF2FD', s=100, label='Retrieved Documents')
    plt.scatter([], [], c='#D0E8C5', s=100, label=f"Post Rerank || n = {n}")
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='lower right', bbox_to_anchor=(1.05, 1))
    plt.suptitle("Retrieval Methods Comparison")
    plt.tight_layout()
    if save_fig:
        plt.save_fig(f'plots/retrieval_methods_comparison_k{k}.png')
    plt.show()


def generate_rename_dict(query_entities, retrieved_docs):
    """
    Generate a dictionary that maps document IDs to labels including the count of shared entities
    between the query and the document.

    Args:
        query_entities (dict): The entities extracted from the query.
        retrieved_docs (list): A list of retrieved documents with metadata.

    Returns:
        dict: A dictionary mapping document IDs to labels in the format:
              {doc_id: "doc_id:<num_of_shared_entities> shared entities"}.
    """
    rename_dict = {}
    query_entities_set = set()

    for key, values in query_entities.items():
        if key != "text" and values:
            if isinstance(values, list):
                query_entities_set.update(values)
            else:
                query_entities_set.add(values)
    for doc in retrieved_docs:
        doc_entities = set()
        for key, values in doc["metadata"].items():
            if isinstance(values, list):
                doc_entities.update(values)
            elif values:
                doc_entities.add(values)

        shared_entities = query_entities_set.intersection(doc_entities)
        rename_dict[doc["id"]] = len(shared_entities)
    return rename_dict


def plot_graphs_comparison10(query_entities, G1, ranked_docs1, G2, ranked_docs2, k, n, save_fig=False,
                             rerank_title="Entities Reranking Retrieval", rename_dict=None):
    """
    Plot two graphs side-by-side and highlight the top n ranked documents with special markers.
    Shared nodes between G1 and G2 will have an outer red line. Allows dynamic relabeling and subplot framing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [1, 1]})

    # fig, axes = plt.subplots(2, 1)

    def get_node_styles(G, ranked_docs, rename_dict, shared_edges):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        node_colors = ['#D0E8C5' if node in top_n_ids else '#DDF2FD' for node in G.nodes()]
        node_colors = ["#427D9D" if rename_dict.get(node, 0) != 0 else nc for nc, node in zip(node_colors, G.nodes())]
        edges_colors = ["#61677A" if edge in shared_edges else "#D8D9DA" for edge in G.edges()]
        node_sizes = [400 if node in top_n_ids else 400 for node in G.nodes()]
        return node_colors, node_sizes, edges_colors

    def get_outer(G, ranked_docs):
        top_n_ids = {doc['id'] for doc in ranked_docs[:n]}
        outer_sizes = [450 if node in top_n_ids else 0 for node in G.nodes()]
        outer_colors = ['#FF0000' if node in special_nodes else '#FFFFFF' for node in G.nodes()]
        return outer_sizes, outer_colors

    shared_nodes = set(G1.nodes()).intersection(set(G2.nodes()))
    shared_edges = set(G1.edges()).intersection(set(G2.edges()))
    rename_dict = generate_rename_dict(query_entities, ranked_docs1)
    node_colors1, node_sizes1, edges_colors1 = get_node_styles(G1, ranked_docs1, rename_dict, shared_edges)
    node_colors2, node_sizes2, edges_colors2 = get_node_styles(G2, ranked_docs2, rename_dict, shared_edges)

    pos1 = nx.spring_layout(G1, seed=42,)# k=0.5)
    pos2 = nx.spring_layout(G2, seed=42,)# k=0.5)

    def get_updated_labels(G, rename_dict):
        if rename_dict:
            rename_dict = generate_rename_dict(query_entities, ranked_docs1)
            return {node: rename_dict.get(node, node) for node in G.nodes()}
        return {node: node for node in G.nodes()}

    def draw_graph(ax, G, pos, node_colors, node_sizes, edges_colors):
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edges_colors,
            font_size=8,
            font_color="#000000",
            ax=ax,
        )
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="#000000", ax=ax)

        for node, (x, y) in pos.items():
            if node in shared_nodes:
                circ = plt.Circle((x, y), 0.05, edgecolor='#821131', facecolor='none', linewidth=2,
                                  transform=ax.transData)
                ax.add_patch(circ)

    draw_graph(axes[0], G1, pos1, node_colors1, node_sizes1, edges_colors1)
    # axes[0].text(3, 8, f"Shared entities: {}", style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    axes[0].set_title(rerank_title, fontsize=12, fontweight='bold', pad=20)

    draw_graph(axes[1], G2, pos2, node_colors2, node_sizes2, edges_colors2)
    axes[1].set_title("Basic Retrieval", fontsize=12, fontweight='bold', pad=20)

    for ax in axes:
        rect = patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            edgecolor='black', linewidth=2, facecolor='none'
        )
        ax.add_patch(rect)
    plt.scatter([], [], c='#821131', s=100, label='Retrieved in both methods')
    plt.scatter([], [], c='#DDF2FD', s=100, label='Retrieved Documents')
    plt.scatter([], [], c='#D0E8C5', s=100, label=f"Post Rerank || n = {n}")
    plt.scatter([], [], c='#427D9D', s=100, label=f"Sharing Entities with original query")
    plt.legend(
        scatterpoints=1, frameon=True, labelspacing=1, loc='lower center',
        bbox_to_anchor=(0.2, -0.85), ncol=4
    )

    plt.suptitle("Retrieval Methods Comparison", fontsize=15, fontweight='bold', y=0.95)

    plt.tight_layout()  # (pad=3)
    if save_fig:
        plt.savefig(f'plots/retrieval_methods_comparison_k{k}.png')
    plt.show()
