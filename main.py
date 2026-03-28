import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from data_processing import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from automl import *
from search_space import *
# from visualization import *

import warnings
warnings.filterwarnings("ignore")

def create_pipeline(num_cols, cat_cols, args):
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
    TOP_K = top_k_generate(feature_amount, 10)
    for k in TOP_K:
        dag.add_node(f"top_k_{k}", DiscreteNode(f"top_k_{k}", k))
    
    # feature selection nodes
    for feature_preprocessor in FEATURE_PREPROCESSORS:
        dag.add_node(feature_preprocessor, DiscreteNode(f"sk_feature_preprocessor_{feature_preprocessor}", FEATURE_PREPROCESSORS[feature_preprocessor]))
    
    # model nodes
    for model in MODELS_CLASSIFIERS:
        dag.add_node(model, DiscreteNode(f"sk_model_{model}", MODELS_CLASSIFIERS[model]))

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
        for model in MODELS_CLASSIFIERS.keys():
            dag.add_edge(f"top_k_{k}", model)

    # Layer 5: model nodes to end node
    for model in MODELS_CLASSIFIERS.keys():
        dag.add_edge(model, 'end')
    
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
    data.drop(columns=["id"], inplace=True)
    
    if args.dropna:
        data.dropna(inplace=True)
        
    X, y = handle_target_column(data, args.target)
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Perform ML Pipeline Optimization
    dag = create_pipeline(num_cols, cat_cols, args)
    optimizer = ACOOptimizer(dag, n_ants=50, iterations=50)
    best_pipeline, best_score = optimizer.optimize(X, y, verbose=True, scoring="f1_macro")
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