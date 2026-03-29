import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import joblib
from data_processing import *
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import os
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from automl import *
from search_space import *
# from visualization import *

# Aliases for backward compatibility
feature_selections = FEATURE_SELECTIONS
models_classifiers = MODELS_CLASSIFIERS

import warnings
warnings.filterwarnings("ignore")

def detect_task_type(y):
    """
    Automatically detect whether the task is classification or regression.
    
    Args:
        y: Target variable (pandas Series)
    
    Returns:
        'classification' or 'regression'
    """
    # Check dtype: object, bool, or category -> classification
    if y.dtype == 'object' or y.dtype == 'bool' or str(y.dtype) == 'category':
        return 'classification'
    
    # Check unique values
    unique_count = y.nunique()
    total_count = len(y)
    
    # Classification if: few unique values OR low unique ratio
    if unique_count <= 10 or (unique_count / total_count < 0.05):
        return 'classification'
    
    # Otherwise, treat as regression
    return 'regression'


def create_pipeline(num_cols, cat_cols, args, task_type='classification', model_dict=None):
    """
    Create the DAG pipeline with task-specific configurations.
    
    Args:
        num_cols: List of numeric column names
        cat_cols: List of categorical column names
        args: Command-line arguments
        task_type: 'classification' or 'regression'
        model_dict: Dictionary of models to use (MODELS_CLASSIFIERS or MODELS_REGRESSORS)
    """
    dag = PipelineGraph()
    
    # Use default model dict if not provided
    if model_dict is None:
        model_dict = MODELS_CLASSIFIERS
    
    # Get feature selections based on task type (LDA only for classification)
    feature_selections = get_feature_selections(task_type)
    
    # === Node Setup ===
    dag.add_node('start', DiscreteNode('VirtualStart', None))
    dag.add_node('end', DiscreteNode('VirtualEnd', None))
    
    # Create preprocessors with prefixed IDs to prevent collision
    preprocessors = {}
    for scaler in SCALERS:
        preprocessors[scaler] = tramsform_column(
            num_cols, cat_cols, 
            args.num_fill_strategy, SCALERS[scaler], 
            args.cat_fill_strategy, args.add_poly
        )
        dag.add_node(f"preprocessor_{scaler}", 
                     DiscreteNode(f"Preprocessor_{scaler}", preprocessors[scaler]))
    
    # Create TOP_K nodes with prefixed IDs
    for k in TOP_K:
        dag.add_node(f"topk_{k}", 
                     DiscreteNode(f"Top_k_{k}", TOP_K[k]))
    
    # Create feature selection nodes with prefixed IDs (task-specific)
    for feature_selection in feature_selections:
        dag.add_node(f"feature_{feature_selection}", 
                     DiscreteNode(f"FeatureSelection_{feature_selection}", feature_selections[feature_selection]))
    
    # Create SMOTE node if requested and classification
    if args.use_smote and task_type == 'classification':
        dag.add_node('smote', DiscreteNode('SMOTE', SMOTE()))
    
    # Create model nodes with prefixed IDs (task-specific)
    for model in model_dict:
        dag.add_node(f"model_{model}", 
                     DiscreteNode(f"Model_{model}", model_dict[model]))
    
    # Create hyperparameter nodes (optional - allows ants to choose specific params)
    for model in model_dict:
        params = get_model_params(model)
        for param_key, param_value in params.items():
            dag.add_node(f"param_{param_key}", 
                         DiscreteNode(f"Param_{param_key}", param_value))
    
    # === Strict Layered Edges ===
    # Layer 0 -> Layer 1: start -> preprocessors
    for scaler in SCALERS:
        dag.add_edge('start', f"preprocessor_{scaler}")
    
    # Layer 1 -> Layer 2: preprocessors -> feature_selection (or preprocessors -> SMOTE -> feature_selection)
    for scaler in SCALERS:
        if args.use_smote and task_type == 'classification':
            dag.add_edge(f"preprocessor_{scaler}", 'smote')
            for feature_selection in feature_selections:
                dag.add_edge('smote', f"feature_{feature_selection}")
        else:
            for feature_selection in feature_selections:
                dag.add_edge(f"preprocessor_{scaler}", f"feature_{feature_selection}")
    
    # Layer 2 -> Layer 3: feature_selection -> TOP_K
    for feature_selection in feature_selections:
        for k in TOP_K:
            dag.add_edge(f"feature_{feature_selection}", f"topk_{k}")
    
    # Layer 3 -> Layer 4: TOP_K -> models
    for k in TOP_K:
        for model in model_dict:
            dag.add_edge(f"topk_{k}", f"model_{model}")
    
    # Layer 4 -> Layer 5: models -> hyperparameter nodes (OPTIONAL)
    for model in model_dict:
        params = get_model_params(model)
        # Direct edge: Model -> End (use default params)
        dag.add_edge(f"model_{model}", 'end')
        # Optional edges: Model -> Param -> End (use specific params)
        for param_key in params:
            dag.add_edge(f"model_{model}", f"param_{param_key}")
            dag.add_edge(f"param_{param_key}", 'end')
    
    return dag



EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)
def main(args):
    # Create experiment directory
    experiment_path = EXPERIMENT_DIR / args.name
    experiment_path.mkdir(exist_ok=True)
    config = {
        "target": args.target,
        "task": args.task,
        "dropna": args.dropna,
        "num_fill_strategy": args.num_fill_strategy,
        "cat_fill_strategy": args.cat_fill_strategy,
        "add_poly": args.add_poly,
        "use_smote": args.use_smote
    }
    
    # load data
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path, sep=None, engine='python')
    # sample data if too large to speed up testing (remove for full runs)
    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
    
    if args.dropna:
        data.dropna(inplace=True)
        
    X, y = handle_target_column(data, args.target)
    
    # Determine task type and configure accordingly
    task_type_input = args.task.lower() if args.task else 'auto'
    
    # Auto-detect task type if 'auto' is specified
    if task_type_input == 'auto':
        detected_task = detect_task_type(y)
        task_type = detected_task
        print(f"Auto-detected task type: {task_type}")
    elif 'regression' in task_type_input:
        task_type = 'regression'
    else:
        task_type = 'classification'
    
    # Assign model dict and label encoder based on task type
    if task_type == 'regression':
        model_dict = MODELS_REGRESSORS
        apply_label_encoder = False
    else:
        model_dict = MODELS_CLASSIFIERS
        apply_label_encoder = True
    
    # LabelEncoder ONLY for classification (fixes XGBoost class label error)
    if apply_label_encoder:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Perform ML Pipeline Optimization with task-specific configuration
    dag = create_pipeline(num_cols, cat_cols, args, task_type=task_type, model_dict=model_dict)
    optimizer = ACOOptimizer(dag, n_ants=args.n_ants, iterations=args.iterations, top_k=args.top_k)
    best_pipeline, best_score, top_k_pipelines = optimizer.optimize(X, y, verbose=True, task_type=task_type)
    print(best_pipeline)
    print(f"Best Score: {best_score:.4f}")
    
    # ========================================
    # Results Export and Visualization
    # ========================================
    try:
        # Save config
        with open(os.path.join(experiment_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        # Save best model
        joblib.dump(best_pipeline, os.path.join(experiment_path, "model.pkl"))
        
        # Single Model Evaluation
        best_pipeline.fit(X, y)
        y_pred_single = best_pipeline.predict(X)
        single_metrics = evaluate_comprehensive(y, y_pred_single, task_type, best_score, experiment_path)
        
        # Pheromone Visualization
        try:
            optimizer.visualize_pheromones(os.path.join(experiment_path, 'pheromone_heatmap.png'))
            optimizer.visualize_pheromone_network(os.path.join(experiment_path, 'pheromone_network.png'))
        except Exception as e:
            print(f"Visualization failed: {e}")
        
        # Ensemble Evaluation (if at least 2 unique pipelines)
        ensemble_metrics = None
        ensemble = None
        if top_k_pipelines and len(top_k_pipelines) >= 2:
            ensemble = create_ensemble(top_k_pipelines, task_type)
            if ensemble:
                try:
                    ensemble.fit(X, y)
                    y_pred_ensemble = ensemble.predict(X)
                    ensemble_metrics = evaluate_comprehensive(y, y_pred_ensemble, task_type, best_score, experiment_path)
                except Exception as e:
                    print(f"Ensemble evaluation failed: {e}")
                    ensemble_metrics = None
                # Export protection: Try to save ensemble, but don't crash if it fails
                try:
                    joblib.dump(ensemble, os.path.join(experiment_path, "ensemble_model.pkl"))
                except Exception as e:
                    print(f"Ensemble export failed: {e}")
        
        # Unified Report
        print_unified_report(single_metrics, ensemble_metrics, task_type)
        
    except Exception as e:
        print(f"Export failed: {e}")
    
    print(f"\nResults exported to: experiments/{args.name}/")


def create_ensemble(top_pipelines, task_type):
    """Create VotingClassifier or VotingRegressor from top-K pipelines with probability-aware voting."""
    if not top_pipelines or len(top_pipelines) < 2:
        return None
    
    try:
        if task_type == 'classification':
            from sklearn.ensemble import VotingClassifier
            
            # Check if ALL models support predict_proba for soft voting
            all_support_proba = True
            for _, pipe, _ in top_pipelines:
                try:
                    # Try to get the final estimator
                    final_est = pipe[-1][1] if hasattr(pipe, '__getitem__') else None
                    if final_est is not None and not hasattr(final_est, 'predict_proba'):
                        all_support_proba = False
                        break
                except:
                    all_support_proba = False
                    break
            
            # Use soft voting only if ALL models support it
            voting_mode = 'soft' if all_support_proba else 'hard'
            
            estimators = [(f"model_{i}", pipe) for i, (_, pipe, _) in enumerate(top_pipelines)]
            ensemble = VotingClassifier(estimators=estimators, voting=voting_mode)
        else:
            from sklearn.ensemble import VotingRegressor
            estimators = [(f"model_{i}", pipe) for i, (_, pipe, _) in enumerate(top_pipelines)]
            ensemble = VotingRegressor(estimators=estimators)
        
        return ensemble
        
    except Exception as e:
        print(f"Ensemble creation failed: {e}")
        return None


def evaluate_comprehensive(y_true, y_pred, task_type, best_score, save_path):
    """Generate comprehensive evaluation metrics."""
    if task_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix (Accuracy: {metrics["accuracy"]:.4f})')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        return metrics
    else:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
        }
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predicted vs Actual (R2: {r2:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, 'predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        return metrics


def print_unified_report(single_metrics, ensemble_metrics=None, task_type='classification'):
    """Print unified metrics box with null-safety."""
    
    # Null-safe helper
    def safe_val(metrics, key, default='N/A'):
        if metrics is None or key not in metrics:
            return default
        val = metrics[key]
        if val is None:
            return default
        try:
            return f"{val:.4f}"
        except:
            return default
    
    print("\n" + "=" * 65)
    print("               UNIFIED AUTOML EVALUATION REPORT")
    print("=" * 65)
    
    if task_type == 'classification':
        print(f"\n{'Metric':<28} {'Single Best':<15} {'Ensemble':<15}")
        print("-" * 58)
        print(f"{'Accuracy':<28} {safe_val(single_metrics, 'accuracy'):<15} {safe_val(ensemble_metrics, 'accuracy'):<15}")
        print(f"{'Precision (Macro)':<28} {safe_val(single_metrics, 'precision_macro'):<15} {safe_val(ensemble_metrics, 'precision_macro'):<15}")
        print(f"{'Recall (Macro)':<28} {safe_val(single_metrics, 'recall_macro'):<15} {safe_val(ensemble_metrics, 'recall_macro'):<15}")
        print(f"{'F1-Score (Macro)':<28} {safe_val(single_metrics, 'f1_macro'):<15} {safe_val(ensemble_metrics, 'f1_macro'):<15}")
        print(f"{'F1-Score (Weighted)':<28} {safe_val(single_metrics, 'f1_weighted'):<15} {safe_val(ensemble_metrics, 'f1_weighted'):<15}")
    else:
        print(f"\n{'Metric':<28} {'Single Best':<15} {'Ensemble':<15}")
        print("-" * 58)
        print(f"{'R2 Score':<28} {safe_val(single_metrics, 'r2'):<15} {safe_val(ensemble_metrics, 'r2'):<15}")
        print(f"{'RMSE':<28} {safe_val(single_metrics, 'rmse'):<15} {safe_val(ensemble_metrics, 'rmse'):<15}")
        print(f"{'MAE':<28} {safe_val(single_metrics, 'mae'):<15} {safe_val(ensemble_metrics, 'mae'):<15}")
        print(f"{'MAPE (%)':<28} {safe_val(single_metrics, 'mape'):<15} {safe_val(ensemble_metrics, 'mape'):<15}")
    
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for classification")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name or index")
    parser.add_argument("--task", type=str, default="Auto", help="Task type [Auto, Classification, Regression]")
    parser.add_argument("--iterations", type=int, default=10, help="Number of ACO iterations")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants per iteration")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top pipelines for ensemble")
    parser.add_argument("--dropna", type=bool, default=False, help="Drop NaN values")
    parser.add_argument("--num_fill_strategy", type=str, default="mean", help="Numerical fill strategy [mean, median, most_frequent]")
    parser.add_argument("--cat_fill_strategy", type=str, default="most_frequent", help="Categorical fill strategy [most_frequent, constant]")
    parser.add_argument("--add_poly", type=bool, default=False, help="Add polynomial features")
    parser.add_argument("--use_smote", type=bool, default=False, help="Use SMOTE for oversampling")
    args = parser.parse_args()
    main(args)