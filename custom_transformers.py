# Core modules
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class CPUSeriesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        patterns = [
            r'(Core [im]\d)', 
            r'(Ryzen \d)',
            r'(Atom [xX]\d)',
            r'(Celeron)',
            r'(Pentium)',
            r'(Xeon)',
            r'(A\d+)',
            r'(E2)',
            r'(FX)'
        ]
        
        def extract(cpu_string):
            for pattern in patterns:
                match = re.search(pattern, str(cpu_string), re.IGNORECASE)
                if match:
                    return match.group(1)
            return "Other"
        X["CPU_series"] = X["CPU_model"].apply(extract)
        return X
    
class GPUSeriesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        def extract(gpu_string):
            gpu_string = str(gpu_string)
            gpu_map = [
                (r'GTX', "GeForce GTX"),
                (r'MX\d{3}', "GeForce MX"),
                (r'GeForce \d{3}', "GeForce (Low-end)"),
                (r'Quadro', "Quadro"),
                (r'Radeon Pro', "Radeon Pro"),
                (r'Radeon RX', "Radeon RX"),
                (r'Radeon R\d', "Radeon R"),
                (r'Radeon', "Radeon (Other)"),
                (r'UHD Graphics', "Intel UHD"),
                (r'Iris Plus', "Intel Iris Plus"),
                (r'Iris Pro', "Intel Iris Pro"),
                (r'Iris', "Intel Iris"),
                (r'HD Graphics', "Intel HD")
            ]
            for pattern, label in gpu_map:
                if re.search(pattern, gpu_string, re.IGNORECASE):
                    return label
            return "Other"
        X["GPU_series"] = X["GPU_model"].apply(extract)
        return X
        
class CardinalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, columns, threshold = "mean", exceptions = None):
        self.columns = columns
        self.threshold = threshold
        self.exceptions = exceptions or {} # {col: [values to never group (Apple)]}
        
    def fit(self, X, y = None):
        self.keep_categories_ = {}
        for col in self.columns:
            counts = X[col].value_counts() # count each category
            thresh = counts.mean() if self.threshold == "mean" else self.threshold # calculate threshold
            keep = set(counts[counts >= thresh].index) # categories ABOVE threshold
            if col in self.exceptions:
                keep.update(self.exceptions[col])  # always keep Apple, etc.
            self.keep_categories_[col] = keep
        return self
    
    def _to_others(self, value, keep):
        if value in keep:
            return value
        return "Other"
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            keep = self.keep_categories_[col]
            X[col] = X[col].apply(self._to_others, args = (keep,))
        return X
    
class PixelCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, log_transform = True):
        self.log_transform = log_transform
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["Pixels"] = (X["ScreenW"] * X["ScreenH"]).astype(int)
        if self.log_transform:
            X["Pixels"] = np.log(X["Pixels"])
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = np.log1p(X[col])
        return X
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        drop_cols = [c for c in self.columns if c in X.columns]
        return X.drop(columns = drop_cols)
    
class NumericFeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        self.numeric_cols_ = X.select_dtypes(include = "number").columns.tolist()
        return self
    
    def transform(self, X):
        return X[self.numeric_cols_].copy()

class KMeansClusterAdder(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters = 4, random_state = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def fit(self, X, y = None):
        self.n_features_in_ = X.shape[1]
        self.kmeans_ = KMeans(
            n_clusters = self.n_clusters,
            random_state = self.random_state,
            n_init = 10
        )
        self.kmeans_.fit(X)
        return self
    
    def transform(self, X):    
        clusters = self.kmeans_.predict(X) # We assign the cluster label as a new column
        return np.column_stack((X, clusters)) # Reshape to 2D for the pipeline
    
    def get_feature_names_out(self, input_features = None): # Labels columns after transformation for future use in feature importance & such
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return list(input_features) + ["Cluster"]