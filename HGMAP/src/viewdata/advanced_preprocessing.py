import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP

class AdvancedPreprocessor:
    def __init__(
        self,
        n_components: int = 50,
        use_umap: bool = True,
        use_pca: bool = True,
        use_kernel_pca: bool = True
    ):
        self.n_components = n_components
        self.use_umap = use_umap
        self.use_pca = use_pca
        self.use_kernel_pca = use_kernel_pca
        
        # Initialize transformers
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.quantile_transformer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=1000
        )
        
        # Dimensionality reduction
        if use_pca:
            self.pca = PCA(n_components=n_components)
        if use_kernel_pca:
            self.kernel_pca = KernelPCA(
                n_components=n_components,
                kernel='rbf'
            )
        if use_umap:
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1
            )
            
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing pipeline"""
        # Power transformation
        X_transformed = self.power_transformer.fit_transform(X)
        
        # Quantile transformation
        X_transformed = self.quantile_transformer.fit_transform(X_transformed)
        
        # Dimensionality reduction
        reduced_features = []
        
        if self.use_pca:
            pca_features = self.pca.fit_transform(X_transformed)
            reduced_features.append(pca_features)
            
        if self.use_kernel_pca:
            kpca_features = self.kernel_pca.fit_transform(X_transformed)
            reduced_features.append(kpca_features)
            
        if self.use_umap:
            umap_features = self.umap.fit_transform(X_transformed)
            reduced_features.append(umap_features)
        
        # Combine all features
        if reduced_features:
            X_final = np.hstack(reduced_features)
        else:
            X_final = X_transformed
            
        return X_final 