import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from data_processing import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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

def create_pipeline(num_cols, cat_cols, args):
    dag = PipelineGraph()
    
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
    
    # Create feature selection nodes with prefixed IDs
    for feature_selection in FEATURE_SELECTIONS:
        dag.add_node(f"feature_{feature_selection}", 
                     DiscreteNode(f"FeatureSelection_{feature_selection}", FEATURE_SELECTIONS[feature_selection]))
    
    # Create model nodes with prefixed IDs
    for model in MODELS_CLASSIFIERS:
        dag.add_node(f"model_{model}", 
                     DiscreteNode(f"Model_{model}", MODELS_CLASSIFIERS[model]))
    
    # === Strict Layered Edges ===
    # Layer 0 -> Layer 1: start -> preprocessors
    for scaler in SCALERS:
        dag.add_edge('start', f"preprocessor_{scaler}")
    
    # Layer 1 -> Layer 2: preprocessors -> feature_selection
    for scaler in SCALERS:
        for feature_selection in FEATURE_SELECTIONS:
            dag.add_edge(f"preprocessor_{scaler}", f"feature_{feature_selection}")
    
    # Layer 2 -> Layer 3: feature_selection -> TOP_K
    for feature_selection in FEATURE_SELECTIONS:
        for k in TOP_K:
            dag.add_edge(f"feature_{feature_selection}", f"topk_{k}")
    
    # Layer 3 -> Layer 4: TOP_K -> models
    for k in TOP_K:
        for model in MODELS_CLASSIFIERS:
            dag.add_edge(f"topk_{k}", f"model_{model}")
    
    # Layer 4 -> Layer 5: models -> end
    for model in MODELS_CLASSIFIERS:
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
    data = pd.read_csv(data_path)
    # sample data if too large to speed up testing (remove for full runs)
    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42).reset_index(drop=True)
    
    if args.dropna:
        data.dropna(inplace=True)
        
    X, y = handle_target_column(data, args.target)
    
    # Encode target variable for classification tasks (fixes XGBoost class label error)
    task_type = args.task.lower() if args.task else 'auto'
    if 'classification' in task_type or task_type == 'auto':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Perform ML Pipeline Optimization
    dag = create_pipeline(num_cols, cat_cols, args)
    optimizer = ACOOptimizer(dag, n_ants=100, iterations=30)
    best_pipeline, best_score = optimizer.optimize(X, y, verbose=True)
    print(f"Best Pipeline: {best_pipeline}")
    print(f"Best Score: {best_score}")
    
    # # save model
    # joblib.dump(model, experiment_path / "model.pkl")
    
    # # Visualize Results
    # plt.figure(figsize=(10, 7))
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    # disp.plot()
    # plt.savefig(experiment_path / "confusion_matrix.png")
    
    # # Save Results
    # print(classification_report(y_test, y_pred), file=open(experiment_path / "classification_report.txt", "w"))
    
    # # save config
    # with open(experiment_path / "config.json", "w") as f:
    #     json.dump(config, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for classification")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name or index")
    parser.add_argument("--task", type=str, default="Auto", help="Task type [Auto, Classification, Regression]")
    parser.add_argument("--dropna", type=bool, default=False, help="Drop NaN values")
    parser.add_argument("--num_fill_strategy", type=str, default="mean", help="Numerical fill strategy [mean, median, most_frequent]")
    parser.add_argument("--cat_fill_strategy", type=str, default="most_frequent", help="Categorical fill strategy [most_frequent, constant]")
    parser.add_argument("--add_poly", type=bool, default=False, help="Add polynomial features")
    parser.add_argument("--use_smote", type=bool, default=False, help="Use SMOTE for oversampling")
    args = parser.parse_args()
    main(args)