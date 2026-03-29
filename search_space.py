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
TOP_K = {
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
    "logistic": lambda: LogisticRegression(max_iter=2000),
    'rf': RandomForestClassifier,
    'svc': SVC,
    'knn': KNeighborsClassifier,
    'nb': GaussianNB,
    'dt': DecisionTreeClassifier,
    'ada': AdaBoostClassifier,
    'gbm': GradientBoostingClassifier,
    'xgb': XGBClassifier,
    'mlp': MLPClassifier,
    'lda': LinearDiscriminantAnalysis,
    'qda': lambda: QuadraticDiscriminantAnalysis(reg_param=0.1),
}

MODELS_REGRESSORS = {
    "lr": lambda: LinearRegression(),
    "ridge": lambda: Ridge(),
    "lasso": lambda: Lasso(),
    "elastic": lambda: ElasticNet(),
    "rf": RandomForestRegressor,
    "svr": SVR,
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