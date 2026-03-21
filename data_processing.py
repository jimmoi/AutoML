import pandas as pd
import numpy as np
import json
import joblib

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from typing import Literal

def handle_target_column(df, target=None):
    try:
        if isinstance(target, str):
            y = df[target]
            X = df.drop(columns=[target])
        elif isinstance(target, int):
            y = df.iloc[:, target]
            X = df.drop(df.columns[target], axis=1)
        return X, y
    except Exception as e:
        print(e)
        raise ValueError("target must be column name or index. Or target column is not found")

def tramsform_column(numerical_columns, categorical_column, num_fill_strategy, cat_fill_strategy, add_poly):
    
    # Numeric pipeline
    num_steps = []
    num_steps.append(("impute", SimpleImputer(strategy=num_fill_strategy)))
    if add_poly:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    num_pipeline = Pipeline(num_steps)
    
    # Categorical pipeline
    cat_steps = []
    cat_steps.append(("impute", SimpleImputer(strategy=cat_fill_strategy)))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    cat_pipeline = Pipeline(cat_steps)
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_column)
        ])
    
    return preprocessor

def preprocess_data(
    data:pd.DataFrame,
    path_to_save: str,
    target_column: str,
    nan_strategy: Literal["drop", "fill"] = "fill",
    num_fill_strategy: Literal["mean", "median", "most_frequent"] = "mean",
    cat_fill_strategy: Literal["most_frequent", "constant"] = "most_frequent",
    add_poly: bool = False,
    use_smote: bool = False,
    ):
    df = data.copy()

    # Handle NaN
    if nan_strategy == "drop":
        df.dropna(inplace=True)
    
    # Handle target column
    X, y = handle_target_column(df, target_column)
    
    # Detect column types
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Preprocessing
    preprocessor = tramsform_column(num_cols, cat_cols, num_fill_strategy, cat_fill_strategy, add_poly)
    X = preprocessor.fit_transform(X)
    
    # SMOTE
    if use_smote:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
    
    # Save config
    config = {
        "target": str(target_column),
        "nan_strategy": nan_strategy,
        "num_fill_strategy": num_fill_strategy,
        "cat_fill_strategy": cat_fill_strategy,
        "use_smote": use_smote,
        "add_poly": add_poly
    }
        
    # Save preprocessor
    joblib.dump(preprocessor, path_to_save / "preprocessor.pkl")
    
    return X, y, preprocessor, config
    
    
if __name__ == "__main__":
    pass
    
    
    
    
    