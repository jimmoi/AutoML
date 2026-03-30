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

def tramsform_column(numerical_columns, categorical_columns, num_fill_strategy, num_scale, cat_fill_strategy, add_poly):
    
    # Numeric pipeline
    num_steps = []
    num_steps.append(("impute", SimpleImputer(strategy=num_fill_strategy)))
    
    # Instantiate the scaler class - sklearn Pipeline requires instances, not classes
    if num_scale is not None:
        num_steps.append(("scale", num_scale()))  # Fixed: instantiate the class
        
    if add_poly:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    num_pipeline = Pipeline(num_steps)
    
    # Build transformers list dynamically based on available columns
    transformers = [("num", num_pipeline, numerical_columns)]
    
    # Only add categorical pipeline if there are categorical columns
    if len(categorical_columns) > 0:
        # print(f"Categorical columns detected: {categorical_columns}")
        # กัน strategy พัง
        if cat_fill_strategy not in ["most_frequent", "constant"]:
            cat_fill_strategy = "most_frequent"

        # กัน sklearn version
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                max_categories=20,
                min_frequency=5
            )
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore")

        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy=cat_fill_strategy)),
            ("onehot", encoder)
        ])

        transformers.append(("cat", cat_pipeline, categorical_columns))
    
    # Combine pipelines dynamically
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any unspecified columns
    )
    
    return preprocessor

if __name__ == "__main__":
    pass
    
    
    
    
    