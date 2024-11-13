import json
import os
from collections import defaultdict
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio


def load_graph_data(entities_file, edges_file):
    """
    Load the graph data from entities and edges JSON files.
    :param entities_file:
    :param edges_file:
    :return:
    """
    with open(entities_file, 'r') as f:
        entity_weights = json.load(f)

    with open(edges_file, 'r') as f:
        edges = json.load(f)

    return entity_weights, edges


def create_filtered_networkx_graph(entity_weights, edges, edge_weight_threshold=6):
    """
    Create a NetworkX graph with filtered edges and nodes.
    Only include edges with a weight greater than the specified threshold.
    :param entity_weights:
    :param edges:
    :param edge_weight_threshold:
    :return:
    """
    G = nx.Graph()

    # Add filtered edges and nodes
    for edge in edges:
        entity1 = edge['entity1']
        entity2 = edge['entity2']
        weight = edge['weight']

        # Only include edges with a weight greater than the threshold
        if weight > edge_weight_threshold:
            # Add the nodes (entities) with weights
            if not G.has_node(entity1):
                G.add_node(entity1, weight=entity_weights[entity1])
            if not G.has_node(entity2):
                G.add_node(entity2, weight=entity_weights[entity2])

            # Add the edge with its weight and associated chunk IDs
            G.add_edge(entity1, entity2, weight=weight, chunks=edge['chunks'])

    return G


def visualize_filtered_entity_graph(G, file_path='entity_graph.html'):
    """
    Visualize the filtered entity graph using Plotly and save it as an HTML file.
    The size of nodes will represent their weight, and the thickness of edges will represent the edge weight.
    Node and edge information will be displayed only on hover.
    :param G:
    :param file_path:
    :return:
    """
    # Increase 'k' value and iterations to spread out the graph further
    pos = nx.spring_layout(G, seed=42, k=2, iterations=500)  # Increased 'k' for more spreading

    # Create Plotly scatter plot for nodes
    node_x = []
    node_y = []
    node_sizes = []
    node_texts = []

    # Adjust the scaling of node sizes
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_size = G.nodes[node]['weight'] * 0.3  # Further reduce node size scaling
        node_size = min(node_size, 30)  # Limit maximum node size
        node_sizes.append(node_size)
        node_texts.append(f"Entity: {node}, Weight: {G.nodes[node]['weight']}")  # Hover text

    edge_traces = []

    # Create edges in the plot
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Adjust edge width scaling factor
        edge_width = edge[2]['weight'] / 5  # Reduced scaling factor for edges
        # Add detailed hover information for edges (weight and chunks)
        edge_text = f"Entities: {edge[0]} - {edge[1]}\nCo-occurrences: {edge[2]['weight']}\nChunks: {', '.join(map(str, edge[2]['chunks']))}"

        # Create individual scatter plot for each edge with its width
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=edge_width, color='#888'),  # Assign the width per edge
            hoverinfo='text',
            text=edge_text,  # Hover text for the edge
            mode='lines'
        )

        # Add the edge trace to the figure data
        edge_traces.append(edge_trace)

    # Create the node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_texts,  # Text that appears on hover
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_sizes,
            color=node_sizes,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
            line_width=2
        )
    )

    # Create the figure with nodes and edges
    fig = go.Figure(data=edge_traces + [node_trace],  # Display nodes and edges together
                    layout=go.Layout(
                        title='Filtered Entity-Graph Visualization',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        dragmode='lasso'  # Enable moving nodes individually
                    ))

    # Show the figure in the notebook or browser
    fig.show()

    # Save the graph as an interactive HTML file
    pio.write_html(fig, file=file_path, auto_open=False)  # Save the figure without auto-opening


def load_entities(file_path):
    """Load the entity-tagged data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def create_entity_graph(entities_data):
    """
    Create an entity graph based on contextual proximity.
    Exclude "AGE" and "SEX" entities. Entities are nodes, and edges represent their co-occurrence.
    """
    entity_weights = defaultdict(int)  # To count occurrences of each entity (node weight)
    edge_weights = defaultdict(lambda: defaultdict(int))  # To count co-occurrences (edge weights)

    # Iterate through each data chunk
    for entry in entities_data:
        chunk_id = entry['id']
        entity_types = entry.keys()

        # Gather all entities from this chunk, excluding "AGE" and "SEX"
        entities_in_chunk = []
        for entity_type in entity_types:
            if entity_type not in ['text', 'id', 'AGE', 'SEX']:  # Exclude AGE and SEX entities
                entities_in_chunk.extend(entry[entity_type])

        # Update node weights (entity occurrences)
        for entity in entities_in_chunk:
            entity_weights[entity] += 1

        # Update edge weights (co-occurrence within the same chunk)
        for i in range(len(entities_in_chunk)):
            for j in range(i + 1, len(entities_in_chunk)):
                entity1 = entities_in_chunk[i]
                entity2 = entities_in_chunk[j]
                if entity1 != entity2:
                    edge_weights[(entity1, entity2)][chunk_id] += 1

    return entity_weights, edge_weights


def save_graph(entity_weights, edge_weights, output_dir):
    """Save the entities and edges as separate JSON files."""
    # Save entities with weights
    entities_file = os.path.join(output_dir, 'entities.json')
    with open(entities_file, 'w') as f:
        json.dump(entity_weights, f, indent=4)
    print(f"Entities saved to {entities_file}")

    # Save edges with their chunk connections
    edges_file = os.path.join(output_dir, 'edges.json')
    formatted_edges = []

    for (entity1, entity2), chunks in edge_weights.items():
        formatted_edges.append({
            'entity1': entity1,
            'entity2': entity2,
            'chunks': list(chunks.keys()),
            'weight': sum(chunks.values())  # Sum of co-occurrences
        })

    with open(edges_file, 'w') as f:
        json.dump(formatted_edges, f, indent=4)
    print(f"Edges saved to {edges_file}")


def main():
    # Load the entity-tagged data from both files
    data_dir = '../data'
    output_dir = 'output'

    # Check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

        medmcqa_data = load_entities(os.path.join(data_dir, 'medmcqa/train_entities.json'))
        medqa_data = load_entities(os.path.join(data_dir, 'medqa/train_entities.json'))

        combined_data = medmcqa_data + medqa_data
        # Create the entity graph (without AGE and SEX entities)
        entity_weights, edge_weights = create_entity_graph(combined_data)

        save_graph(entity_weights, edge_weights, output_dir)

    # Load the entities and edges data
    output_dir = 'output'
    entities_file = f"{output_dir}/entities.json"
    edges_file = f"{output_dir}/edges.json"

    entity_weights, edges = load_graph_data(entities_file, edges_file)

    # Create a NetworkX graph with filtered edges 
    G = create_filtered_networkx_graph(entity_weights, edges, edge_weight_threshold=3)

    # Visualize the filtered entity graph using Plotly
    visualize_filtered_entity_graph(G)


if __name__ == "__main__":
    main()
