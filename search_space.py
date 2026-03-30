from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

SCALERS = {
    "none": None, 
    "standard": StandardScaler, 
    "minmax": MinMaxScaler, 
    "robust": RobustScaler, 
    "normalizer": Normalizer
}

FEATURE_SELECTIONS = {
    "none": None, 
    "pca": PCA, 
    "selectkbest": SelectKBest, 
    "variancethreshold": VarianceThreshold, 
    # LDA is only for classification - conditionally added in main.py
}

# TOP_K = (1,100) # percentage of features to keep
# TOP_K = {
#     "1": 1,
#     "2": 2,
#     "3": 3,
#     "5": 5,
#     "10":10,
#     "15":15,
#     "20":20,
# }

TOP_K = {
    "3":0.03,
    "5":0.05,
    "10":0.1,
    "20":0.2,
    "30":0.3,
    "40":0.4,
    "50":0.5,
    "60":0.6,
    "70":0.7,
    "80":0.8,
    "90":0.9,
    "100":1.0
}

IMBALANCED_TECHNIQUES = {
    "none": None, 
    # "smote": SMOTE
}

MODELS_CLASSIFIERS = {
    "logistic": LogisticRegression,
    'rf': RandomForestClassifier,
    'svc': lambda: SVC(probability=True),  # Enable probability for soft voting
    'knn': KNeighborsClassifier,
    'nb': GaussianNB,
    'dt': DecisionTreeClassifier,
    'ada': AdaBoostClassifier,
    'gbm': GradientBoostingClassifier,
    'xgb': XGBClassifier,
    'mlp': MLPClassifier,
    'lda': LinearDiscriminantAnalysis,
    'qda': QuadraticDiscriminantAnalysis,
}

MODELS_REGRESSORS = {
    "lr": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic": ElasticNet,
    "rf": RandomForestRegressor,
    "svr": SVR,
}

MODEL_PARAMS = {
    "logistic": {
        "log_c01": {"C": 0.1, "max_iter": 2000},
        "log_c1": {"C": 1.0, "max_iter": 2000},
        "log_c100": {"C": 100.0, "max_iter": 2000},
    },
    "rf": {
        "rf_n100": {"n_estimators": 100},
        "rf_n200": {"n_estimators": 200},
        "rf_n500": {"n_estimators": 500},
        "rf_d10": {"max_depth": 10},
        "rf_d20": {"max_depth": 20},
    },
    "svc": {
        "svc_c1": {"C": 1.0, "probability": True},
        "svc_c10": {"C": 10.0, "probability": True},
        "svc_rbf": {"kernel": "rbf", "probability": True},
    },
    "knn": {
        "knn_5": {"n_neighbors": 5},
        "knn_10": {"n_neighbors": 10},
    },
    "dt": {
        "dt_d5": {"max_depth": 5},
        "dt_d10": {"max_depth": 10},
    },
    "xgb": {
        "xgb_d3": {"max_depth": 3},
        "xgb_d6": {"max_depth": 6},
        "xgb_lr01": {"learning_rate": 0.1},
        "xgb_lr001": {"learning_rate": 0.01},
    },
    "mlp": {
        "mlp_100": {"hidden_layer_sizes": (100,)},
        "mlp_100_100": {"hidden_layer_sizes": (100, 100)},
    },
    "gbm": {
        "gbm_d3": {"max_depth": 3},
        "gbm_d5": {"max_depth": 5},
    },
    "lda": {
        "lda_2": {"n_components": 2},
    },
    "qda": {
        "qda_reg": {"reg_param": 0.1},
    },
    "lr": {
        "lr_a1": {"alpha": 1.0},
        "lr_a01": {"alpha": 0.1},
    },
    "ridge": {
        "ridge_a1": {"alpha": 1.0},
        "ridge_a10": {"alpha": 10.0},
    },
    "lasso": {
        "lasso_a001": {"alpha": 0.01},
        "lasso_a1": {"alpha": 1.0},
    },
    "elastic": {
        "elastic_01": {"l1_ratio": 0.1},
        "elastic_05": {"l1_ratio": 0.5},
    },
    "svr": {
        "svr_rbf": {"kernel": "rbf"},
        "svr_linear": {"kernel": "linear"},
    },
}

def get_feature_selections(task_type='classification'):
    """Get feature selections dict, including LDA only for classification."""
    fs = {
        "none": None, 
        "pca": PCA, 
        "selectkbest": SelectKBest, 
        "variancethreshold": VarianceThreshold, 
    }
    if task_type == 'classification':
        fs["lda"] = LinearDiscriminantAnalysis
    return fs

def get_model_params(model_key):
    """Get hyperparameters for a specific model."""
    return MODEL_PARAMS.get(model_key, {})