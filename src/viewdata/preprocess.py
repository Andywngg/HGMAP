import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats import entropy
import config.constants as constants
from robust_clr import robustclr
from shap_feature_selector import SHAPFeatureSelector

class MicrobiomePreprocessor:
    def __init__(self, data):
        self.data = data
        self.processed_data = None
    
    def clean_data(self):
        # Remove rows with all zeros or NaNs
        self.data = self.data.dropna(how='all')
        self.data = self.data[~(self.data == 0).all(axis=1)]
        return self
    
    def normalize_abundance(self):
        # Normalize to relative abundance
        self.data = self.data.div(self.data.sum(axis=1), axis=0)
        return self
    
    def compute_diversity_metrics(self):
        # Compute diversity metrics
        shannon_index = self.data.apply(lambda x: entropy(x[x > 0]), axis=1)
        richness = (self.data > 0).sum(axis=1)
        evenness = shannon_index / np.log(richness)
        
        return shannon_index, richness, evenness
    
    def feature_engineering(self):
        # Log normalization
        data_log = np.log1p(self.data)
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = pd.DataFrame(imputer.fit_transform(data_log), columns=self.data.columns)
        
        # Diversity metrics
        shannon_index, richness, evenness = self.compute_diversity_metrics()
        
        # Polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(data_imputed)
        
        # PCA
        pca = PCA(n_components=10)
        pca_features = pca.fit_transform(poly_features)
        
        # Combine features
        data_final = pd.DataFrame(pca_features)
        data_final["Shannon_Index"] = shannon_index
        data_final["Richness"] = richness
        data_final["Evenness"] = evenness
        
        # Standardize
        scaler = StandardScaler()
        self.processed_data = pd.DataFrame(
            scaler.fit_transform(data_final), 
            columns=data_final.columns
        )
        
        return self
    
    def save_processed_data(self, filename="processed_microbiome_data.csv"):
        if self.processed_data is not None:
            self.processed_data.to_csv(filename, index=False)
        else:
            raise ValueError("Data not processed yet. Call feature_engineering() first.")
    
    def advanced_preprocessing(self, data):
        # Compositional Data Analysis
        data_clr = robustclr(data)  # Robust CLR transformation
        
        # Advanced Feature Engineering
        features = {
            'taxonomic': self._compute_taxonomic_features(data),
            'functional': self._compute_functional_features(data),
            'diversity': self._compute_diversity_metrics(data),
            'network': self._compute_network_features(data),
            'temporal': self._compute_temporal_features(data)
        }
        
        # Feature Selection using SHAP
        selector = SHAPFeatureSelector(
            n_features=100,
            importance_threshold=0.01
        )
        selected_features = selector.fit_transform(features)
        
        return selected_features

# Usage
from loaddata import taxonomy_data
preprocessor = MicrobiomePreprocessor(taxonomy_data)
preprocessor.clean_data().normalize_abundance().feature_engineering().save_processed_data()