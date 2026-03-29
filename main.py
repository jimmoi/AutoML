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
    mean_squared_error
)
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
    
    # Layer 4 -> Layer 5: models -> end
    for model in model_dict:
        dag.add_edge(f"model_{model}", 'end')
    
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
    optimizer = ACOOptimizer(dag, n_ants=args.n_ants, iterations=args.iterations)
    best_pipeline, best_score = optimizer.optimize(X, y, verbose=True, task_type=task_type)
    print(best_pipeline)
    print(f"Best Score: {best_score:.4f}")
    
    # ========================================
    # Results Export and Visualization
    # ========================================
    try:
        joblib.dump(best_pipeline, experiment_path / "model.pkl")
        
        with open(experiment_path / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        best_pipeline.fit(X, y)
        y_pred = best_pipeline.predict(X)
        
        if task_type == 'classification':
            plt.figure(figsize=(10, 7))
            cm = confusion_matrix(y, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f"Confusion Matrix - Best Score: {best_score:.4f}")
            plt.savefig(experiment_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            report = classification_report(y, y_pred)
            with open(experiment_path / "metrics_report.txt", "w") as f:
                f.write("=" * 50 + "\n")
                f.write("CLASSIFICATION METRICS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Best Score (Accuracy): {best_score:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write("-" * 50 + "\n")
                f.write(report)
        else:
            plt.figure(figsize=(10, 7))
            plt.scatter(y, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
            
            min_val = min(min(y), min(y_pred))
            max_val = max(max(y), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Predicted vs Actual - R2: {best_score:.4f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(experiment_path / "predicted_vs_actual.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            with open(experiment_path / "metrics_report.txt", "w") as f:
                f.write("=" * 50 + "\n")
                f.write("REGRESSION METRICS REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Best Score (R2): {best_score:.4f}\n")
                f.write(f"R2 Score: {r2:.4f}\n")
                f.write(f"RMSE: {rmse:.4f}\n")
            
    except Exception as e:
        print(f"Export failed: {e}")
    
    print(f"\nResults exported to: experiments/{args.name}/")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for classification")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name or index")
    parser.add_argument("--task", type=str, default="Auto", help="Task type [Auto, Classification, Regression]")
    parser.add_argument("--iterations", type=int, default=10, help="Number of ACO iterations")
    parser.add_argument("--n_ants", type=int, default=20, help="Number of ants per iteration")
    parser.add_argument("--dropna", type=bool, default=False, help="Drop NaN values")
    parser.add_argument("--num_fill_strategy", type=str, default="mean", help="Numerical fill strategy [mean, median, most_frequent]")
    parser.add_argument("--cat_fill_strategy", type=str, default="most_frequent", help="Categorical fill strategy [most_frequent, constant]")
    parser.add_argument("--add_poly", type=bool, default=False, help="Add polynomial features")
    parser.add_argument("--use_smote", type=bool, default=False, help="Use SMOTE for oversampling")
    args = parser.parse_args()
    main(args)