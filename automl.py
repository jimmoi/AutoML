import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class PipelineNode:
    """Represents a specific step in the ML pipeline (e.g., PCA, SVM)."""
    def __init__(self, name, component, params_space=None):
        self.name = name
        self.component = component  # The sklearn-like class
        self.params_space = params_space or {}

class PipelineGraph:
    """Manages the DAG of possible pipeline configurations."""
    def __init__(self):
        self.adj_list = {}
        self.nodes = {}
        self.pheromones = {} # Edges to pheromone levels

    def add_node(self, node_id, node_obj):
        self.nodes[node_id] = node_obj
        self.adj_list[node_id] = []

    def add_edge(self, from_id, to_id, initial_pheromone=0.1):
        self.adj_list[from_id].append(to_id)
        self.pheromones[(from_id, to_id)] = initial_pheromone

    def get_successors(self, node_id):
        return self.adj_list.get(node_id, [])

class ACOOptimizer:
    def __init__(self, graph, n_ants=10, iterations=20, alpha=1.0, beta=2.0, decay=0.1):
        self.graph = graph
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance (skipped here for simplicity)
        self.decay = decay

    def _select_next_node(self, current_node):
        successors = self.graph.get_successors(current_node)
        if not successors:
            return None
        
        
        probs = []# Calculate transition probabilities based on pheromones
        for s in successors:
            tau = self.graph.pheromones[(current_node, s)]
            probs.append(tau ** self.alpha)
        
        total = sum(probs)
        probs = [p / total for p in probs]
        return np.random.choice(successors, p=probs)

    def optimize(self, X, y, preprocessor=None, scoring='accuracy', verbose=False):
        best_pipeline = None
        best_score = -np.inf

        for i in range(self.iterations):
            ant_results = []
            
            for _ in range(self.n_ants):
                path = self._construct_path()
                score, pipeline = self._evaluate_path(path, X, y, scoring, preprocessor)
                ant_results.append((path, score))
                
                if score > best_score:
                    best_score = score
                    best_path = path
                    best_pipeline = pipeline

            self._update_pheromones(ant_results)
            if verbose:
                print(f"Iteration {i+1}: Best Score = {best_score:.4f}")
                print(f"Best current Pipeline: {best_path}")

        return best_pipeline, best_score

    def _construct_path(self):
        path = ['start'] # Assume a virtual start node
        current = 'start'
        while True:
            nxt = self._select_next_node(current)
            if nxt is None: break
            path.append(nxt)
            current = nxt
        return path[1:-1] # Remove 'start' and 'end'

    def _evaluate_path(self, path, X, y, scoring, preprocessor):
        # Build sklearn pipeline from path
        steps = []
        if preprocessor:
            steps.append(("preprocessor", preprocessor))
        for node_id in path:
            node = self.graph.nodes[node_id]
            # In a full ACO(R) implementation, you would also sample 
            # hyperparameters from node.params_space here.
            steps.append((node.name, node.component()))
        
        try:
            clf = Pipeline(steps)
            scores = cross_val_score(clf, X, y, cv=3, scoring=scoring, n_jobs=-1)
            return scores.mean() ,clf
        except:
            return 0 ,None # Return poor score for invalid paths

    def _update_pheromones(self, results):
        # Evaporation
        for edge in self.graph.pheromones:
            self.graph.pheromones[edge] *= (1 - self.decay)
        
        # Reinforcement
        for path, score in results:
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                if edge in self.graph.pheromones:
                    self.graph.pheromones[edge] += score
                    

