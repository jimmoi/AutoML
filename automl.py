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
    def __init__(self, graph, n_ants=5, iterations=5, alpha=0.6, beta=4.0, decay=0.2):
        self.graph = graph
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance (skipped here for simplicity)
        self.decay = decay
        self.pipeline_cache = {}  # Cache for evaluated paths
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_heuristic(self, node_id):
        """
        Calculate heuristic value for a node based on its type.
        Higher values indicate more promising components.
        
        Args:
            node_id: The node identifier string
            
        Returns:
            float: Heuristic value between 0 and 1
        """
        # Model nodes: favor strong performers
        if 'model_' in node_id:
            strong_models = ['xgb', 'rf', 'gbm']
            medium_models = ['svc', 'mlp']
            
            for model in strong_models:
                if model in node_id:
                    return 0.8
            for model in medium_models:
                if model in node_id:
                    return 0.5
            return 0.3
        
        # Feature selection nodes: favor effective selectors
        if 'feature_' in node_id:
            effective_selectors = ['selectkbest', 'pca']
            for selector in effective_selectors:
                if selector in node_id:
                    return 0.7
            return 0.4
        
        # Hyperparameter nodes: neutral heuristic
        if 'param_' in node_id:
            return 0.5
        
        # All other nodes: neutral heuristic
        return 0.5
    
    def _select_next_node(self, current_node):
        successors = self.graph.get_successors(current_node)
        if not successors:
            return None
        
        # Calculate transition probabilities using ACO formula:
        # probability = (pheromone^alpha) * (heuristic^beta)
        probs = []
        for s in successors:
            tau = self.graph.pheromones[(current_node, s)]
            heuristic = self._get_heuristic(s)
            probs.append((tau ** self.alpha) * (heuristic ** self.beta))
        
        # Safeguard against division by zero
        total = sum(probs)
        if total == 0:
            probs = [1/len(probs) for _ in probs]
        else:
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
        pending_model_class = None  # Track model class for param combination
        pending_model_name = None  # Track model name
        collected_params = {}  # Collect hyperparameters from param nodes
        
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
            
            # Handle model nodes - store for later param combination
            if 'model_' in node_id:
                model_class = node.value if hasattr(node, 'value') and node.value else node.value
                if callable(model_class):
                    pending_model_class = model_class
                    pending_model_name = node.name
                    collected_params = {}
                else:
                    # No params expected - add directly to steps
                    steps.append((node.name, model_class))
                    pending_model_class = None
                continue
            
            # Handle hyperparameter nodes - collect params
            if 'param_' in node_id:
                if pending_model_class is not None and hasattr(node, 'value'):
                    param_dict = node.value if isinstance(node.value, dict) else {}
                    if param_dict:
                        collected_params.update(param_dict)
                continue
            
            # Handle preprocessor nodes - append directly
            if 'preprocessor_' in node_id:
                preprocessor = node.value if hasattr(node, 'value') and node.value else node.value
                if callable(preprocessor):
                    steps.append((node.name, preprocessor))
                continue
        
        # Add model with collected params (if any)
        if pending_model_class is not None and pending_model_name is not None:
            try:
                if collected_params:
                    model_instance = pending_model_class(**collected_params)
                else:
                    model_instance = pending_model_class()
                steps.append((pending_model_name, model_instance))
            except Exception:
                # Fallback to default
                try:
                    steps.append((pending_model_name, pending_model_class()))
                except Exception:
                    pass
        
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
            
            # Use pheromone-based selection with heuristic (alpha + beta)
            probs = []
            for s in valid_successors:
                tau = self.graph.pheromones.get((current, s), 0.1)
                heuristic = self._get_heuristic(s)
                probs.append((tau ** self.alpha) * (heuristic ** self.beta))
            
            # Safeguard against division by zero
            total = sum(probs)
            if total == 0:
                probs = [1/len(probs) for _ in probs]
            else:
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
        Uses caching to avoid redundant evaluations of identical paths.
        
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
        # Create hashable cache key from path
        path_key = tuple(path)
        
        # Check cache first
        if path_key in self.pipeline_cache:
            self.cache_hits += 1
            return self.pipeline_cache[path_key]
        
        self.cache_misses += 1
        steps = self._decode_path(path)
        
        try:
            # Dynamically select scoring metric based on task type
            if task_type == 'classification':
                effective_scoring = 'accuracy'
            elif task_type == 'regression':
                effective_scoring = 'r2'
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
                # R2 score ranges from 0-1, higher is better.
                # Use max(0, score) since R2 can be negative for very poor models.
                fitness = max(0.0, mean_score)
            else:
                fitness = mean_score
            
            # Cache the result (including fitted pipeline)
            self.pipeline_cache[path_key] = (fitness, clf)
            return fitness, clf
            
        except Exception as e:
            # Cache failed evaluations to prevent redundant retries
            self.pipeline_cache[path_key] = (0.0, None)
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


