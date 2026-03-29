from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import f_classif

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

SCALERS = {
    "scaler_none": None, 
    "scaler_standard": StandardScaler, 
    "scaler_minmax": MinMaxScaler, 
    "scaler_robust": RobustScaler, 
    "scaler_normalizer": Normalizer
}

def prepare_feature_preprocessor(feature_preprocessor):
    match feature_preprocessor:
        case "pca":
            def wrapper(k):
                return PCA(n_components=k)
            return wrapper
        case "selectkbest":
            def wrapper(k):
                return SelectKBest(score_func=f_classif, k=k)
            return wrapper
        case "variancethreshold":
            def wrapper(k):
                return VarianceThreshold(threshold=0.01)
            return wrapper
        case "lda":
            def wrapper(k):
                return LinearDiscriminantAnalysis(n_components=min(k, len(set(y)) - 1))
            return wrapper
    return wrapper

FEATURE_PREPROCESSORS = {
    "feature_preprocessor_none": None, 
    "feature_preprocessor_pca": prepare_feature_preprocessor("pca"), 
    "feature_preprocessor_selectkbest": prepare_feature_preprocessor("selectkbest"), 
    # "feature_preprocessor_variancethreshold": prepare_feature_preprocessor("variancethreshold"), # !Error
    # "feature_preprocessor_lda": prepare_feature_preprocessor("lda") # !Error
}

IMBALANCED_TECHNIQUES = {
    "imbalanced_none": None, 
    # "imbalanced_smote": SMOTE
}

MODELS_CLASSIFIERS = {
    "model_logistic": LogisticRegression,
    "model_rf": RandomForestClassifier,
    "model_svc": SVC,
    "model_knn": KNeighborsClassifier,
    "model_nb": GaussianNB,
    "model_dt": DecisionTreeClassifier,
    "model_ada": AdaBoostClassifier,
    "model_gbm": GradientBoostingClassifier,
    # "model_xgb": XGBClassifier, # !Error y encoder
    "model_mlp": MLPClassifier,
    "model_lda": LinearDiscriminantAnalysis,
    # "model_qda": QuadraticDiscriminantAnalysis,  # !Error highly collinear (redundant)
}

MODELS_REGRESSION = {
    "model_rf": RandomForestRegressor,
    "model_svc": SVR,
    "model_knn": KNeighborsRegressor,
    "model_dt": DecisionTreeRegressor,
    "model_ada": AdaBoostRegressor,
    "model_gbm": GradientBoostingRegressor,
    # "model_xgb": XGBRegressor, # !Error y encoder
    "model_mlp": MLPRegressor,
    "model_linear": LinearRegression,
    "model_ridge": Ridge,
    "model_lasso": Lasso,
    "model_elasticnet": ElasticNet
}

MODELS_PARAMS = {
    "model_logistic": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["lbfgs", "liblinear", "saga"]
    },
    "model_rf": {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "model_svc": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto", 0.01, 0.1, 1.0]
    },
    "model_knn": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "model_nb": {
        "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
    },
    "model_dt": {
        "max_depth": [None, 10, 20, 30, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "model_ada": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 1.0, 2.0]
    },
    "model_gbm": {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0]
    },
    "model_xgb": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.3],
        "max_depth": [3, 5, 7],
        "subsample": [0.6, 0.8, 1.0]
    },
    "model_mlp": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["tanh", "relu"],
        "solver": ["sgd", "adam"],
        "alpha": [0.0001, 0.001, 0.01]
    },
    "model_lda": {
        "solver": ["svd", "lsqr", "eigen"]
    },
    "model_qda": {
        "reg_param": [0.0, 0.1, 0.5, 0.9]
    },
    "model_linear": {
        "fit_intercept": [True, False]
    },
    "model_ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    },
    "model_lasso": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "selection": ["cyclic", "random"]
    },
    "model_elasticnet": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "selection": ["cyclic", "random"]
    }
}