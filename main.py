import pandas as pd
import numpy as np
from pathlib import Path
import argparse
# from visualization import *
# from data_processing import *


def main(args):
    print("start")
    print(args)
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # data = pd.read_csv(data_path)
    # print(data.head())
    print("pass")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML for classification")
    parser.add_argument("--name", type=str, default="untitled", help="Name of the experiment")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--preprocess", type=bool, default=False, help="Preprocess the dataset")
    args = parser.parse_args()
    main(args)