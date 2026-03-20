import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE


def preprocess_pipeline(
    data,
    target=None,
    y_input=None,

    nan_strategy=None,          # None | "drop" | "fill"
    num_fill_strategy="mean",   # numeric
    cat_fill_strategy="most_frequent",  # categorical

    scaling=None,
    feature_method=None,
    k=5,
    use_smote=False,
    add_poly=False,

    save_config_path=None,
    save_preprocessor_path=None
):
    # ------------------------
    # 1. Convert data
    # ------------------------
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

    # ------------------------
    # 2. Handle target
    # ------------------------
    if y_input is not None:
        X = df.copy()
        y = pd.Series(y_input)
    else:
        if target is None:
            target = "target"

        if isinstance(target, str):
            y = df[target]
            X = df.drop(columns=[target])
        elif isinstance(target, int):
            y = df.iloc[:, target]
            X = df.drop(df.columns[target], axis=1)
        else:
            raise ValueError("target must be column name or index")

    # ------------------------
    # 3. Handle NaN (drop)
    # ------------------------
    if nan_strategy == "drop":
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined.iloc[:, :-1]
        y = combined.iloc[:, -1]

    # ------------------------
    # 4. Detect column types
    # ------------------------
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # ------------------------
    # 5. Numeric pipeline
    # ------------------------
    num_steps = []

    if nan_strategy == "fill":
        num_steps.append(("impute", SimpleImputer(strategy=num_fill_strategy)))
    else:
        num_steps.append(("impute", SimpleImputer(strategy="mean")))

    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scaling == "normalize":
        num_steps.append(("scaler", Normalizer()))

    if add_poly:
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

    num_pipeline = Pipeline(num_steps)

    # ------------------------
    # 6. Categorical pipeline
    # ------------------------
    if len(cat_cols) > 0:
        if nan_strategy == "fill":
            cat_imputer = SimpleImputer(strategy=cat_fill_strategy)
        else:
            cat_imputer = SimpleImputer(strategy="most_frequent")

        cat_pipeline = Pipeline([
            ("impute", cat_imputer),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    else:
        cat_pipeline = "drop"

    # ------------------------
    # 7. Combine
    # ------------------------
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # ------------------------
    # 8. Apply preprocessing
    # ------------------------
    X = preprocessor.fit_transform(X)

    # ------------------------
    # 9. Feature selection / extraction
    # ------------------------
    if feature_method == "kbest":
        selector = SelectKBest(score_func=f_classif, k=k)
        X = selector.fit_transform(X, y)

    elif feature_method == "variance":
        selector = VarianceThreshold(threshold=0.01)
        X = selector.fit_transform(X)

    elif feature_method == "pca":
        pca = PCA(n_components=k)
        X = pca.fit_transform(X)

    elif feature_method == "lda":
        lda = LinearDiscriminantAnalysis(n_components=min(k, len(set(y)) - 1))
        X = lda.fit_transform(X, y)

    # ------------------------
    # 10. SMOTE
    # ------------------------
    if use_smote:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    # ------------------------
    # 11. Save config
    # ------------------------
    config = {
        "target": str(target),
        "nan_strategy": nan_strategy,
        "num_fill_strategy": num_fill_strategy,
        "cat_fill_strategy": cat_fill_strategy,
        "scaling": scaling,
        "feature_method": feature_method,
        "k": k,
        "use_smote": use_smote,
        "add_poly": add_poly
    }

    if save_config_path:
        os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
        with open(save_config_path, "w") as f:
            json.dump(config, f, indent=4)

    # ------------------------
    # 12. Save preprocessor
    # ------------------------
    if save_preprocessor_path:
        os.makedirs(os.path.dirname(save_preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, save_preprocessor_path)

    # ------------------------
    # 13. Output
    # ------------------------
    print("Final shape:", X.shape)
    print("Class distribution:\n", pd.Series(y).value_counts())

    return X, y, preprocessor