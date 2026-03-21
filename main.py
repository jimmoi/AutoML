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
# from automl import *
# from visualization import *

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
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    if args.preprocess:
        X_train, y_train, preprocessor, preprocessor_config = preprocess_data(df_train, experiment_path, args.target)
        config.update(preprocessor_config)
    else:
        X_train, y_train = handle_target_column(df_train, args.target)
    
    
        
    X_test, y_test = handle_target_column(df_test, args.target)
    X_test = preprocessor.transform(X_test)
    
    # # Perform ML Pipeline Optimization
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    
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