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