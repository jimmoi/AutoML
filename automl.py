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
    def __init__(self, graph, n_ants=5, iterations=5, alpha=1.0, beta=2.0, decay=0.1):
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

    def optimize(self, X, y, scoring='accuracy', verbose=False, task_type='classification'):
        best_pipeline = None
        best_score = -np.inf
        best_path = None
        failed_count = 0  # Track failed pipeline evaluations

        for i in range(self.iterations):
            ant_results = []
            
            for _ in range(self.n_ants):
                path = self._construct_path()
                score, pipeline = self._evaluate_path(path, X, y, scoring, task_type)
                ant_results.append((path, score))
                
                if score == 0.0:
                    failed_count += 1
                
                if score > best_score:
                    best_score = score
                    best_path = path
                    best_pipeline = pipeline

            self._update_pheromones(ant_results)
            if verbose:
                print(f"Iteration {i+1}: Best Score = {best_score:.4f}")
                print(f"Best current Pipeline: {best_path}")

        if verbose and failed_count > 0:
            print(f"[INFO] Total failed pipelines: {failed_count}/{self.n_ants * self.iterations}")

        return best_pipeline, best_score
    
    def _decode_path(self, path):
        """
        Decode a path into sklearn Pipeline steps.
        
        Handles the merging of feature_selection + TOP_K into a single transformer step.
        Different feature selectors accept parameters differently:
        - PCA: uses n_components (float 0-1)
        - SelectKBest: replaced with SelectPercentile (percentile)
        - VarianceThreshold/others: use default params
        - None: skip feature selection
        """
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold
        
        steps = []
        path_params = {}
        pending_feature_selector = None  # Track pending feature selection
        
        for node_id in path:
            node = self.graph.nodes[node_id]
            
            # Skip virtual nodes
            if node_id in ('start', 'end') or node.value is None:
                continue
            
            # Handle feature_selection nodes - store for later combination with TOP_K
            if 'feature_' in node_id:
                selector_class = node.value() if callable(node.value) else node.value
                selector_name = node.name
                
                # If selector is None ("none" option), don't pending - just skip
                if selector_class is None:
                    pending_feature_selector = None  # Clear any pending
                else:
                    pending_feature_selector = (selector_name, selector_class)
                continue  # Don't add yet - wait for TOP_K
            
            # Handle TOP_K nodes - merge with pending feature selector
            if 'topk_' in node_id:
                topk_value = node.value  # Float like 0.9
                
                if pending_feature_selector is not None:
                    selector_name, selector_class = pending_feature_selector
                    
                    # Case 4: Selector is None ("none" feature selection)
                    if selector_class is None:
                        pass  # Skip feature selection entirely
                    
                    # Case 1: PCA - use n_components parameter
                    elif selector_class == PCA or (callable(selector_class) and selector_class.__name__ == 'PCA'):
                        try:
                            # PCA accepts n_components as float (0-1) for variance ratio
                            steps.append((selector_name, PCA(n_components=topk_value)))
                        except Exception:
                            # Fallback to default
                            steps.append((selector_name, PCA()))
                    
                    # Case 2: SelectKBest - replace with SelectPercentile
                    elif selector_class == SelectKBest or (callable(selector_class) and 'SelectKBest' in selector_class.__name__):
                        try:
                            percentile = int(topk_value * 100)
                            steps.append((selector_name, SelectPercentile(percentile=percentile)))
                        except Exception:
                            # Fallback to default SelectKBest
                            steps.append((selector_name, SelectKBest()))
                    
                    # Case 3: VarianceThreshold or other selectors - use default params
                    else:
                        try:
                            # Instantiate with default parameters
                            if callable(selector_class):
                                steps.append((selector_name, selector_class()))
                            else:
                                pass  # Unknown selector, skip
                        except Exception:
                            pass  # Skip if instantiation fails
                    
                    pending_feature_selector = None  # Reset
                continue
            
            # Handle model nodes - append directly
            if 'model_' in node_id:
                model_class = node.value() if callable(node.value) else node.value
                if model_class is not None:
                    steps.append((node.name, model_class))
                continue
            
            # Handle preprocessor nodes - append directly
            if 'preprocessor_' in node_id:
                preprocessor = node.value() if callable(node.value) else node.value
                if preprocessor is not None:
                    steps.append((node.name, preprocessor))
                continue
        
        return steps

    def _construct_path(self):
        path = ['start']
        visited = {'start'}
        current = 'start'
        
        while True:
            successors = self.graph.get_successors(current)
            # Filter out already visited nodes to prevent loops and duplicates
            valid_successors = [s for s in successors if s not in visited]
            
            if not valid_successors:
                break
            
            # Use pheromone-based selection from valid successors only
            probs = []
            for s in valid_successors:
                tau = self.graph.pheromones.get((current, s), 0.1)
                probs.append(tau ** self.alpha)
            
            total = sum(probs)
            probs = [p / total for p in probs]
            nxt = np.random.choice(valid_successors, p=probs)
            
            path.append(nxt)
            visited.add(nxt)
            current = nxt
            
            # Stop if we reached 'end'
            if current == 'end':
                break
        
        return path

    def _evaluate_path(self, path, X, y, scoring='accuracy', task_type='classification'):
        """
        Evaluate a pipeline path with dynamic scoring based on task type.
        
        Args:
            path: List of node IDs representing the pipeline path
            X: Feature data
            y: Target data
            scoring: Base scoring metric (overridden by task_type)
            task_type: 'classification' or 'regression'
        
        Returns:
            tuple: (fitness_score, fitted_pipeline)
                   fitness_score is always positive (higher is better)
                   Returns (0.0, None) on failure
        """
        steps = self._decode_path(path)
        
        try:
            # Dynamically select scoring metric based on task type
            if task_type == 'classification':
                effective_scoring = 'accuracy'
            elif task_type == 'regression':
                effective_scoring = 'neg_root_mean_squared_error'
            else:
                effective_scoring = scoring
            
            clf = Pipeline(steps)
            scores = cross_val_score(clf, X, y, cv=5, scoring=effective_scoring, n_jobs=1)
            mean_score = scores.mean()
            
            # Calculate fitness: ensure positive values, higher is better
            if task_type == 'classification':
                # Accuracy is already bounded [0,1], higher is better
                fitness = mean_score
            elif task_type == 'regression':
                # Convert neg_mse to positive fitness (lower RMSE = higher fitness)
                fitness = 1.0 / (1.0 + abs(mean_score))
            else:
                fitness = mean_score
            
            return fitness, clf
            
        except Exception as e:
            # Only print rare/uncommon errors to reduce console littering
            # Common errors (LinAlgError, ConvergenceWarning) are silently ignored
            error_str = str(e).lower()
            uncommon_errors = ['invalid', 'unavailable', 'could not']
            if any(ue in error_str for ue in uncommon_errors):
                print(f"[DEBUG] Pipeline Failed: {e} | Path: {path}")
            return 0.0, None

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

if __name__ == "__main__":
    Test_node = DiscreteNode("test", None)


