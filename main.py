import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import pickle
import json
from automl import *
from search_space import *
from data_processing import *
from visualization import *

import warnings
warnings.filterwarnings("ignore")

def create_pipeline(num_cols, cat_cols, args, limit_n_feature=10):
    #----------------
    # 1. Setup Graph
    #----------------
    
    dag = PipelineGraph()
    
    #----------------------
    # 2. Define Components
    #----------------------
    
    preprocessors = {}
    for scaler in SCALERS:
        preprocessors[scaler] = tramsform_column(num_cols, cat_cols, args.num_fill_strategy, SCALERS[scaler], args.cat_fill_strategy, args.add_poly)

    #----------------
    # 3. Setup Nodes
    #----------------
    
    # virtual nodes
    dag.add_node('start', DiscreteNode('VirtualStart', None))
    dag.add_node('end', DiscreteNode('VirtualEnd', None))
    
    # feature scaling nodes
    for scaler in preprocessors:
        dag.add_node(scaler, DiscreteNode(f"sk_preprocessor_{scaler}", preprocessors[scaler]))
    
    # K-hyperparameter nodes
    feature_amount = len(num_cols) + len(cat_cols)
    def top_k_generate(n_column, n_choice):
        func2 = lambda x: np.ceil(n_column*(1e3**((x/x.max())-1)))
        test = np.arange(1,n_choice+1)
        test[1:] = func2(test)[:-1]
        return set(test.tolist())
    TOP_K = top_k_generate(feature_amount, limit_n_feature)
    for k in TOP_K:
        dag.add_node(f"top_k_{k}", DiscreteNode(f"top_k_{k}", k))
    
    # feature selection nodes
    for feature_preprocessor in FEATURE_PREPROCESSORS:
        dag.add_node(feature_preprocessor, DiscreteNode(f"sk_feature_preprocessor_{feature_preprocessor}", FEATURE_PREPROCESSORS[feature_preprocessor]))
    
    # imbalanced technique nodes
    if args.use_smote:
        for imbalanced_technique in IMBALANCED_TECHNIQUES:
            dag.add_node(imbalanced_technique, DiscreteNode(f"sk_imbalanced_technique_{imbalanced_technique}", IMBALANCED_TECHNIQUES[imbalanced_technique]))
    
    # model nodes
    MODELS = MODELS_CLASSIFIERS if args.task == "classification" else MODELS_REGRESSION
    for model in MODELS:
        dag.add_node(model, DiscreteNode(f"sk_model_{model}", MODELS[model], params_space=MODELS_PARAMS.get(model, {})))

    #------------------------------
    # 4. Define valid paths (Edges)
    #------------------------------

    # Layer 1: start to feature scaling nodes
    for preprocessor in preprocessors.keys():
            dag.add_edge('start', preprocessor)

    # Layer 2: feature scaling nodes to feature selection or extraction node
    for preprocessor in preprocessors.keys():
        for feature_preprocessor in FEATURE_PREPROCESSORS.keys():
            dag.add_edge(preprocessor, feature_preprocessor)

    # Layer 3 : feature selection to k-hyperparameter nodes
    for feature_preprocessor in FEATURE_PREPROCESSORS.keys():
        for k in TOP_K:
            dag.add_edge(feature_preprocessor, f"top_k_{k}")

    # Layer 4: k-hyperparameter nodes to model nodes
    for k in TOP_K:
        if args.use_smote:
            for imbalanced_technique in IMBALANCED_TECHNIQUES:
                dag.add_edge(f"top_k_{k}", imbalanced_technique)
            
            for imbalanced_technique in IMBALANCED_TECHNIQUES:
                for model in MODELS.keys():
                    dag.add_edge(imbalanced_technique, model)
        else:
            for model in MODELS.keys():
                dag.add_edge(f"top_k_{k}", model)
            

    # Layer 5: model nodes to end node
    for model in MODELS.keys():
        dag.add_edge(model, 'end')
    
    return dag



EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)
def main(args):
    # Create experiment directory
    experiment_path = EXPERIMENT_DIR / args.name
    experiment_path.mkdir(exist_ok=True)
    config = {
        "experiment_name": args.name,
        "dataset": args.data,
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
    
    if args.dropna:
        data.dropna(inplace=True)
        
    X, y = handle_target_column(data, args.target)
    if args.task == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Perform ML Pipeline Optimization
    match args.task:
        case "classification":
            objective = "f1_macro"
        case "regression":
            objective = "neg_mean_squared_error"
    config["objective"] = objective
    dag = create_pipeline(num_cols, cat_cols, args, limit_n_feature=5,)
    optimizer = ACOOptimizer(dag, n_ants=30, iterations=25, local_search_iters=5, timeout=100)
    best_pipeline, best_score, best_params, score_history, pheromone_history, optimization_time = optimizer.optimize(X_train, y_train, verbose=True, scoring=objective)
    print(f"Best Pipeline: {best_pipeline}")
    print(f"Best Score: {best_score}")
    print(f"Best Params: {best_params}")
    print(f"Optimization Time: {optimization_time:.2f} seconds")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    
    # save model
    joblib.dump(best_pipeline, experiment_path / "model.pkl")
    
    # with open(experiment_path / "graph.pkl", "wb") as f:
    #     pickle.dump(dag, f)
        
    # Visualize optimization
    plot_objective_value(score_history, experiment_path / "objective_value.png")
    visualize_pheromone(dag, pheromone_history, experiment_path / "optimization_behavior.mp4", experiment_path / "final_graph.png")
    
    # Classification Visualize Results
    if args.task == "classification":
        plt.figure(figsize=(10, 7))
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_pipeline.classes_)
        disp.plot()
        plt.savefig(experiment_path / "confusion_matrix.png")
    
    # Save Results
    text = f"Best Pipeline: {best_pipeline}\n"
    text += f"Best Params: {best_params}\n" if best_params else ""
    text += f"Optimization Time: {optimization_time:.2f} seconds\n"
    text += f"Best Score: {best_score}\n"
    match args.task:
        case "classification":
            text += classification_report(y_test, y_pred)
        case "regression":
            text += f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}\n"
            text += f"R2 Score: {r2_score(y_test, y_pred)}\n"
            text += f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}\n"
            text += f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred)}\n"

    with open(experiment_path / "report.txt", "w") as f:
        f.write(text)
    
    # save config
    with open(experiment_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for classification")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name or index")
    parser.add_argument("--task", type=str, required=True, help="Task type [Classification, Regression]")
    parser.add_argument("--dropna", type=bool, default=False, help="Drop NaN values")
    parser.add_argument("--num_fill_strategy", type=str, default="mean", help="Numerical fill strategy [mean, median, most_frequent]")
    parser.add_argument("--cat_fill_strategy", type=str, default="most_frequent", help="Categorical fill strategy [most_frequent, constant]")
    parser.add_argument("--add_poly", type=bool, default=False, help="Add polynomial features")
    parser.add_argument("--use_smote", type=bool, default=False, help="Use SMOTE for oversampling")
    args = parser.parse_args()
    main(args)