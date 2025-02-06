import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import umap
import logging
from typing import Optional, Union, List, Dict
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced preprocessing for microbiome data with multiple dimensionality reduction techniques."""
    
    def __init__(
        self,
        n_components: int = 50,
        use_umap: bool = True,
        use_pca: bool = True,
        use_kernel_pca: bool = True,
        imputation_strategy: str = 'knn',
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        scale_method: str = 'standard',
        random_state: int = 42
    ):
        """
        Initialize the preprocessor.
        
        Args:
            n_components: Number of components for dimensionality reduction
            use_umap: Whether to use UMAP
            use_pca: Whether to use PCA
            use_kernel_pca: Whether to use Kernel PCA
            imputation_strategy: Strategy for imputing missing values ('knn' or 'mean')
            min_prevalence: Minimum prevalence threshold for features
            min_abundance: Minimum abundance threshold for features
            scale_method: Scaling method ('standard', 'robust', or 'minmax')
            random_state: Random state for reproducibility
        """
        self.n_components = n_components
        self.use_umap = use_umap
        self.use_pca = use_pca
        self.use_kernel_pca = use_kernel_pca
        self.imputation_strategy = imputation_strategy
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.scale_method = scale_method
        self.random_state = random_state
        
        # Initialize components
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kpca = KernelPCA(
            n_components=n_components,
            kernel='rbf',
            random_state=random_state
        )
        self.umap_reducer = umap.UMAP(
            n_components=min(n_components, 100),
            random_state=random_state
        )
        
        # Store feature masks and transformations
        self.feature_mask_ = None
        self.feature_names_ = None
        self.feature_importance_ = None
        
    def _filter_features(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Filter features based on prevalence and abundance thresholds."""
        # Calculate prevalence (proportion of non-zero values)
        prevalence = (X > 0).mean(axis=0)
        
        # Calculate mean abundance
        mean_abundance = X.mean(axis=0)
        
        # Create feature mask
        self.feature_mask_ = (prevalence >= self.min_prevalence) & (mean_abundance >= self.min_abundance)
        
        if feature_names is not None:
            self.feature_names_ = [name for name, keep in zip(feature_names, self.feature_mask_) if keep]
        
        logger.info(f"Filtered features from {X.shape[1]} to {self.feature_mask_.sum()}")
        return X[:, self.feature_mask_]
    
    def _handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values using the specified strategy."""
        if np.any(np.isnan(X)):
            if self.imputation_strategy == 'knn':
                X = self.imputer.fit_transform(X)
            else:  # mean imputation
                X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        return X
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using the specified method."""
        return self.scaler.fit_transform(X)
    
    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction techniques."""
        reduced_features = []
        
        if self.use_pca:
            pca_features = self.pca.fit_transform(X)
            reduced_features.append(pca_features)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        if self.use_kernel_pca:
            kpca_features = self.kpca.fit_transform(X)
            reduced_features.append(kpca_features)
        
        if self.use_umap:
            umap_features = self.umap_reducer.fit_transform(X)
            reduced_features.append(umap_features)
        
        if reduced_features:
            return np.hstack(reduced_features)
        return X
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform the data."""
        logger.info("Starting advanced preprocessing pipeline...")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Filter features
        X = self._filter_features(X, feature_names)
        
        # Scale features
        X = self._scale_features(X)
        
        # Apply dimensionality reduction
        X = self._reduce_dimensions(X)
        
        logger.info(f"Preprocessing complete. Output shape: {X.shape}")
        return X
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using the fitted preprocessor."""
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Apply feature mask
        if self.feature_mask_ is not None:
            X = X[:, self.feature_mask_]
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Apply dimensionality reduction
        if self.use_pca:
            X = self.pca.transform(X)
        if self.use_kernel_pca:
            X = self.kpca.transform(X)
        if self.use_umap:
            X = self.umap_reducer.transform(X)
        
        return X
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        if self.feature_names_ is None or not self.use_pca:
            return None
        
        # Use PCA components as feature importance
        importance = np.abs(self.pca.components_).mean(axis=0)
        return dict(zip(self.feature_names_, importance)) 