import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

class DiscreteNode:
    """Represents a specific step in the ML pipeline (e.g., PCA, SVM)."""
    def __init__(self, name, value, params_space=None):
        self.name = name
        self.value = value
        self.params_space = params_space or {}
        
    def __repr__(self):
        return f"Node({self.name})"

class ContinuousNode(DiscreteNode):
    """Extends PipelineNode to handle continuous hyperparameter sampling."""
    def __init__(self, name, value, param_ranges, archive_size=10):
        super().__init__(name, value)
        self.param_ranges = param_ranges  # e.g., {'C': (0.1, 10.0), 'gamma': (0.01, 1.0)}
        self.archive = [] # List of (params_dict, score)
        self.archive_size = archive_size
        self.q = 0.1 # Locality parameter (smaller = more focused search)

    def sample_parameters(self):
        if not self.archive:
            # Cold start: Random sample from range
            return {k: np.random.uniform(low, high) for k, (low, high) in self.param_ranges.items()}
        
        # Sort archive by score (descending)
        self.archive.sort(key=lambda x: x[1], reverse=True)
        
        # 1. Select a solution from archive to serve as the 'mean'
        # Weights follow a Gaussian distribution over the ranks
        weights = [self._get_weight(r) for r in range(1, len(self.archive) + 1)]
        probabilities = np.array(weights) / sum(weights)
        chosen_idx = np.random.choice(len(self.archive), p=probabilities)
        parent_params, _ = self.archive[chosen_idx]
        
        # 2. Sample around the parent solution
        sampled_params = {}
        for param_name, (low, high) in self.param_ranges.items():
            # Standard deviation is the average distance to other solutions in archive
            sigma = self._calculate_sigma(param_name, chosen_idx)
            val = np.random.normal(parent_params[param_name], sigma)
            sampled_params[param_name] = np.clip(val, low, high)
            
        return sampled_params

    def _get_weight(self, rank):
        # ACO_R Weight function: w_l = (1 / (q * k * sqrt(2 * pi))) * exp(...)
        # Simplified version for ranking:
        k = self.archive_size
        exponent = -((rank - 1)**2) / (2 * (self.q * k)**2)
        return (1.0 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(exponent)

    def _calculate_sigma(self, param_name, chosen_idx):
        # Sigma is the average distance from the chosen solution to all others in the archive
        distances = [abs(self.archive[chosen_idx][0][param_name] - sol[0][param_name]) 
                    for sol in self.archive]
        return self.q * sum(distances) / (len(self.archive) - 1 + 1e-6)

    def update_archive(self, new_params, score):
        # priority queue
        self.archive.append((new_params, score))
        self.archive.sort(key=lambda x: x[1], reverse=True)
        self.archive = self.archive[:self.archive_size]

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

    def optimize(self, X, y, scoring='accuracy', verbose=False):
        best_pipeline = None
        best_path = None
        best_score = -np.inf

        for i in range(self.iterations):
            print(f"Iteration {i+1}/{self.iterations}")
            ant_results = []
            
            for _ in range(self.n_ants):
                path = self._construct_path()
                if verbose:
                    print(f" Path: {path}")
                score, pipeline = self._evaluate_path(path, X, y, scoring)
                ant_results.append((path, score))
                
                if score > best_score:
                    best_score = score
                    best_path = path
                    best_pipeline = pipeline

            self._update_pheromones(ant_results)
            if verbose:
                print(f"Best Score = {best_score:.4f}")
                print(f"Best current Pipeline: {best_path}")
                print(2**best_score)
                print("="*50)

        return best_pipeline, best_score
    
    def _decode_path(self, path):
        steps = []
        top_k = None
        feature_preprocessor_idx = None
        
        i=0
        for node_id in path:
            node = self.graph.nodes[node_id]
            if node.value is None:
                continue
            if isinstance(node, DiscreteNode): 
                match node.name:
                    case _ if node.name.startswith("sk_preprocessor"):
                        steps.append((node.name, node.value))
                        i+=1
                    case _ if node.name.startswith("sk_feature_preprocessor"):
                        feature_preprocessor_idx = i
                        steps.append((node.name, node.value))
                        i+=1
                    case _ if node.name.startswith("top_k"):
                        top_k = node.value
                    case _:
                        steps.append((node.name, node.value()))
                        i+=1
                # case _ if isinstance(node, ContinuousNode):
                #     params = node.sample_parameters()
                #     path_params[node_id] = params
                #     steps.append((node.name, params))
                    
            # In a full ACO(R) implementation, you would also sample 
            # hyperparameters from node.params_space here.
            
        if feature_preprocessor_idx is not None:
            node_name, node_value = steps.pop(feature_preprocessor_idx)
            steps.insert(feature_preprocessor_idx, (node_name, node_value(top_k)))
            
        return steps
    
    def _select_next_node(self, current_node):
        successors = self.graph.get_successors(current_node)
        if not successors:
            return None
        
        probs = [] # Calculate transition probabilities based on pheromones
        for s in successors:
            tau = self.graph.pheromones[(current_node, s)]
            probs.append(tau ** self.alpha)
        
        total = sum(probs)
        probs = [p / total for p in probs]
        return np.random.choice(successors, p=probs)

    def _construct_path(self):
        path = ['start'] # Assume a virtual start node
        current = 'start'
        while True:
            nxt = self._select_next_node(current)
            if nxt is None: break
            path.append(nxt)
            current = nxt
        return path

    def _evaluate_path(self, path, X, y, scoring):
        # Build sklearn pipeline from path
        steps = self._decode_path(path)
        
        try:
            clf = Pipeline(steps)
            scores = cross_val_score(clf, X, y, cv=5, scoring=scoring, n_jobs=-1)
            
            ## for continuous node update archive
            # for node_id in path:
            #     if self.graph.nodes[node_id].value is not None:
            #         self.graph.nodes[node_id].update_archive(path_params[node_id], scores)
            
            return scores.mean() ,clf
        except Exception as e:
            print(steps)
            print(clf)
            print("Invalid path", e)
            raise e
            # return 0 ,None # Return poor score for invalid paths

    def _update_pheromones(self, results):
        # Evaporation
        for edge in self.graph.pheromones:
            self.graph.pheromones[edge] *= (1 - self.decay)
        
        # Reinforcement
        for path, score in results:
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                if edge in self.graph.pheromones:
                    if score < 0:
                        score = 2**score
                    self.graph.pheromones[edge] += score

if __name__ == "__main__":
    Test_node = DiscreteNode("test", None)


