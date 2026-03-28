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

SCALERS = {
    "none": None, 
    "standard": StandardScaler, 
    "minmax": MinMaxScaler, 
    "robust": RobustScaler, 
    "normalizer": Normalizer
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
    "none": None, 
    "pca": prepare_feature_preprocessor("pca"), 
    "selectkbest": prepare_feature_preprocessor("selectkbest"), 
    # "variancethreshold": prepare_feature_preprocessor("variancethreshold"), # !Error
    # "lda": prepare_feature_preprocessor("lda") # !Error
}

IMBALANCED_TECHNIQUES = {
    "none": None, 
    # "smote": SMOTE
}

MODELS_CLASSIFIERS = {
    "logistic": LogisticRegression,
    'rf': RandomForestClassifier,
    'svc': SVC,
    'knn': KNeighborsClassifier,
    'nb': GaussianNB,
    'dt': DecisionTreeClassifier,
    'ada': AdaBoostClassifier,
    'gbm': GradientBoostingClassifier,
    # 'xgb': XGBClassifier, # !Error y encoder
    'mlp': MLPClassifier,
    'lda': LinearDiscriminantAnalysis,
    # 'qda': QuadraticDiscriminantAnalysis,  # !Error highly collinear (redundant)
}
# models_regression = {
#     'rf': {'name': 'RandomForestRegressor', 'component': RandomForestRegressor},
#     'svc': {'name': 'SVR', 'component': SVR}
# }