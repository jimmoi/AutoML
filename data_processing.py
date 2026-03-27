from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

def tramsform_column(numerical_columns, categorical_column, num_fill_strategy, num_scale, cat_fill_strategy, add_poly):
    
    # Numeric pipeline
    num_steps = []
    num_steps.append(("impute", SimpleImputer(strategy=num_fill_strategy)))
    
    if num_scale is not None:
        num_steps.append(("scale", num_scale))
        
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

if __name__ == "__main__":
    pass
    
    
    
    
    