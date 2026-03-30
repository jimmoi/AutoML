import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import copy

def plot_objective_value(score_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(score_history) + 1), score_history, marker='o', linestyle='-', color='b')
    plt.title('Optimization Objective Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Value')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def _get_node_layers(dag):
    layers = {}
    
    # Layer 0: start
    layers['start'] = 0
    
    # Simple BFS to assign layers
    queue = [('start', 0)]
    visited = {'start'}
    
    while queue:
        node, layer = queue.pop(0)
        
        for neighbor in dag.get_successors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                layers[neighbor] = layer + 1
                queue.append((neighbor, layer + 1))
            else:
                # If a node is reached from multiple paths, push it further if needed
                layers[neighbor] = max(layers[neighbor], layer + 1)
                
    return layers

def visualize_pheromone(dag, pheromone_history, video_path, image_path):
    G = nx.DiGraph()
    for u in dag.nodes:
        G.add_node(u)
    for u in dag.adj_list:
        for v in dag.adj_list[u]:
            G.add_edge(u, v)
            
    layers = _get_node_layers(dag)
    
    # Group nodes by layer to calculate positions
    layer_nodes = {}
    for node, layer in layers.items():
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
        
    pos = {}
    max_layer = max(layer_nodes.keys())
    for layer, nodes in layer_nodes.items():
        x = layer / max_layer # Normalize x
        y_step = 1.0 / (len(nodes) + 1)
        for i, node in enumerate(sorted(nodes)):
            y = 1.0 - (i + 1) * y_step # Normalize y, top to bottom
            pos[node] = (x, y)
            
    # Setup the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Node features
    node_size = 300
    node_color = 'skyblue'
    
    def update(frame):
        ax.clear()
        ax.set_title(f"Pheromone Levels - Iteration {frame + 1}")
        
        current_pheromones = pheromone_history[frame]
        
        # Calculate edge weights for width
        if current_pheromones:
            max_p = max(current_pheromones.values())
            min_p = min(current_pheromones.values())
        else:
            max_p = min_p = 1
            
        edge_widths = []
        edge_colors = []
        for u, v in G.edges():
            p = current_pheromones.get((u, v), 0.1)
            # Normalize for coloring and width
            if max_p > min_p:
                norm_p = (p - min_p) / (max_p - min_p + 1e-6)
            else:
                norm_p = 1.0
            edge_widths.append(1 + 5 * norm_p) # Width from 1 to 6
            edge_colors.append(plt.cm.Blues(0.3 + 0.7 * norm_p)) # Avoid completely white edges
            
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=node_color, edgecolors='black')
        
        # Draw labels above or below nodes
        # Add alternating offsets or just a fixed offset above
        label_pos = {k: (v[0], v[1] + 0.04) for k, v in pos.items()}
        # Simplify names for better look
        labels = {}
        for k in G.nodes():
            v = k
            if v.startswith("sk_preprocessor_"): v = v.replace("sk_preprocessor_", "")
            elif v.startswith("sk_feature_preprocessor_"): v = v.replace("sk_feature_preprocessor_", "")
            elif v.startswith("top_k_"): v = v.replace("top_k_", "k=")
            elif v.startswith("sk_imbalanced_technique_"): v = v.replace("sk_imbalanced_technique_", "")
            elif v.startswith("sk_model_"): v = v.replace("sk_model_", "")
            labels[k] = v
            
        nx.draw_networkx_labels(G, label_pos, labels=labels, ax=ax, font_size=8, font_family='sans-serif', font_weight='bold')
        ax.axis('off')
        
    ani = animation.FuncAnimation(fig, update, frames=len(pheromone_history), blit=False)
    
    # Save the video
    try:
        writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='AutoML'), bitrate=1800)
        ani.save(str(video_path), writer=writer)
    except Exception as e:
        print(f"Failed to save MP4 (make sure ffmpeg is installed): {e}")
        print("Saving as GIF fallback...")
        ani.save(str(video_path).replace('.mp4', '.gif'), writer='pillow', fps=5)
        
    # Save final graph image
    update(len(pheromone_history) - 1)
    plt.savefig(image_path)
    plt.close()