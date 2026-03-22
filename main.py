import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from data_processing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from automl import *
# from visualization import *


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def create_pipeline():
    #----------------
    # 1. Setup Graph
    #----------------
    
    dag = PipelineGraph()
    
    #----------------------
    # 2. Define Components
    #----------------------
    
    scalers = {
        'std': {'name': 'StandardScaler', 'component': StandardScaler},
        'minmax': {'name': 'MinMaxScaler', 'component': MinMaxScaler},
        'robust': {'name': 'RobustScaler', 'component': RobustScaler},
        'normal': {'name': 'Normalizer', 'component': Normalizer}
    }
    # feature_selections = {
    #     'pca': {'name': 'PCA', 'component': PCA},
    #     'selectkbest': {'name': 'SelectKBest', 'component': SelectKBest},
    #     'variancethreshold': {'name': 'VarianceThreshold', 'component': VarianceThreshold},
    #     'lda': {'name': 'LinearDiscriminantAnalysis', 'component': LinearDiscriminantAnalysis}
    # }
    models_classifier = {
        "logistic": {'name': 'LogisticRegression', 'component': LogisticRegression},
        'rf': {'name': 'RandomForestClassifier', 'component': RandomForestClassifier},
        'svc': {'name': 'SVC', 'component': SVC},
        'knn': {'name': 'KNeighborsClassifier', 'component': KNeighborsClassifier},
        'nb': {'name': 'GaussianNB', 'component': GaussianNB},
        'dt': {'name': 'DecisionTreeClassifier', 'component': DecisionTreeClassifier},
        'ada': {'name': 'AdaBoostClassifier', 'component': AdaBoostClassifier},
        'gbm': {'name': 'GradientBoostingClassifier', 'component': GradientBoostingClassifier},
        'xgb': {'name': 'XGBClassifier', 'component': XGBClassifier},
        'mlp': {'name': 'MLPClassifier', 'component': MLPClassifier},
        'lda': {'name': 'LinearDiscriminantAnalysis', 'component': LinearDiscriminantAnalysis},
        'qda': {'name': 'QuadraticDiscriminantAnalysis', 'component': QuadraticDiscriminantAnalysis},
    }
    # models_regression = {
    #     'rf': {'name': 'RandomForestRegressor', 'component': RandomForestRegressor},
    #     'svc': {'name': 'SVR', 'component': SVR}
    # }
    
    #----------------
    # 3. Setup Nodes
    #----------------
    
    # virtual nodes
    dag.add_node('start', PipelineNode('Start', None))
    dag.add_node('end', PipelineNode('End', None))
    # feature scaling nodes
    all_component = [scalers, models_classifier]
    for component in all_component:
        for key, value in component.items():
            dag.add_node(key, PipelineNode(value['name'], value['component']))
    
    #------------------------------
    # 4. Define valid paths (Edges)
    #------------------------------
    
    # Layer 1: start to feature scaling nodes
    for scaler in scalers.keys():
        dag.add_edge('start', scaler)
        
    # # Layer 2: feature scaling nodes to feature selection or extraction node
    # dag.add_edge('std', 'pca')
    # dag.add_edge('minmax', 'pca')
    
    # Layer 3: feature selection or extraction node to model nodes
    for scaler in scalers.keys():
        for model in models_classifier.keys():
            dag.add_edge(scaler, model)
            
    # Layer 4: model nodes to end node
    for model in models_classifier.keys():
        dag.add_edge(model, 'end')
    
    return dag



EXPERIMENT_DIR = Path("experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)
def main(args):
    # Create experiment directory
    experiment_path = EXPERIMENT_DIR / args.name
    experiment_path.mkdir(exist_ok=True)
    config = {}
    
    # load data
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    
    # prepare & preprocess data
    if args.preprocess:
        X, y, preprocessor, preprocessor_config = preprocess_data(data, experiment_path, args.target)
        config.update(preprocessor_config)
    else:
        X, y = handle_target_column(data, args.target)
    
    # Perform ML Pipeline Optimization
    dag = create_pipeline()
    optimizer = ACOOptimizer(dag)
    best_pipeline, best_score = optimizer.optimize(X, y, preprocessor, verbose=True)
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
    parser.add_argument("--preprocess", type=bool, default=False, help="Preprocess the dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name or index")
    parser.add_argument("--task", type=str, default="Auto", help="Task type [Auto, Classification, Regression]")
    args = parser.parse_args()
    main(args)