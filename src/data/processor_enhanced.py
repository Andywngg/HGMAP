import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import json

class EnhancedMicrobiomeProcessor:
    def __init__(
        self,
        data_dir: str = "data",
        n_components: int = 50,
        min_prevalence: float = 0.1,
        random_state: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.n_components = n_components
        self.min_prevalence = min_prevalence
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.logger = logging.getLogger(__name__)
        
    def compute_diversity_metrics(self, abundance_matrix):
        """Compute comprehensive diversity metrics."""
        # Convert to proportions
        proportions = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
        
        # Shannon diversity
        shannon = -(proportions * np.log1p(proportions)).sum(axis=1)
        
        # Simpson diversity
        simpson = 1 - (proportions ** 2).sum(axis=1)
        
        # Species richness
        richness = (abundance_matrix > 0).sum(axis=1)
        
        # Pielou's evenness
        evenness = shannon / np.log1p(richness)
        
        # Berger-Parker dominance
        dominance = abundance_matrix.max(axis=1) / abundance_matrix.sum(axis=1)
        
        # Chao1 richness estimator
        singletons = (abundance_matrix == 1).sum(axis=1)
        doubletons = (abundance_matrix == 2).sum(axis=1)
        chao1 = richness + (singletons * (singletons - 1)) / (2 * (doubletons + 1))
        
        # Faith's Phylogenetic Diversity (simplified version)
        pd_faith = np.log1p(abundance_matrix).sum(axis=1)
        
        return pd.DataFrame({
            'shannon_diversity': shannon,
            'simpson_diversity': simpson,
            'species_richness': richness,
            'evenness': evenness,
            'dominance': dominance,
            'chao1': chao1,
            'faith_pd': pd_faith
        })
    
    def compute_interaction_features(self, abundance_matrix):
        """Compute advanced microbial interaction features."""
        # Get top abundant species
        mean_abundance = abundance_matrix.mean()
        top_species = mean_abundance.nlargest(20).index
        
        interaction_features = {}
        
        # Pairwise interactions
        for i, sp1 in enumerate(top_species):
            for sp2 in top_species[i+1:]:
                # Multiplicative interaction
                col_name = f"interaction_{sp1}_{sp2}"
                interaction_features[col_name] = abundance_matrix[sp1] * abundance_matrix[sp2]
                
                # Ratio interaction
                ratio_name = f"ratio_{sp1}_{sp2}"
                interaction_features[ratio_name] = abundance_matrix[sp1] / (abundance_matrix[sp2] + 1e-10)
        
        # Community-level features
        interaction_features['total_abundance'] = abundance_matrix.sum(axis=1)
        interaction_features['abundance_variance'] = abundance_matrix.var(axis=1)
        
        return pd.DataFrame(interaction_features)
    
    def compute_network_features(self, abundance_matrix):
        """Compute network-based features using correlation patterns."""
        correlation_matrix = abundance_matrix.corr()
        
        # Basic network metrics
        degree_centrality = correlation_matrix.abs().mean()
        max_correlation = correlation_matrix.max()
        min_correlation = correlation_matrix.min()
        correlation_std = correlation_matrix.std()
        
        # Clustering coefficient (simplified)
        clustering = (correlation_matrix ** 3).mean()
        
        # Network density
        density = (correlation_matrix.abs() > 0.5).mean()
        
        return pd.DataFrame({
            'degree_centrality': degree_centrality,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'correlation_std': correlation_std,
            'clustering': clustering,
            'network_density': density
        })
    
    def process_abundance_data(self, abundance_df):
        """Process abundance data with advanced feature engineering."""
        try:
            # 1. Handle missing values using KNN imputation
            abundance_matrix = pd.DataFrame(
                self.knn_imputer.fit_transform(abundance_df),
                columns=abundance_df.columns,
                index=abundance_df.index
            )
            
            # 2. Filter rare taxa
            prevalence = (abundance_matrix > 0).mean()
            abundant_taxa = prevalence[prevalence >= self.min_prevalence].index
            abundance_matrix = abundance_matrix[abundant_taxa]
            
            # 3. Log transformation with pseudocount
            abundance_matrix = np.log1p(abundance_matrix)
            
            # 4. Compute diversity metrics
            diversity_features = self.compute_diversity_metrics(abundance_matrix)
            
            # 5. Compute interaction features
            interaction_features = self.compute_interaction_features(abundance_matrix)
            
            # 6. Compute network features
            network_features = self.compute_network_features(abundance_matrix)
            
            # 7. PCA transformation
            pca_features = pd.DataFrame(
                self.pca.fit_transform(abundance_matrix),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                index=abundance_matrix.index
            )
            
            # 8. Combine all features
            all_features = pd.concat([
                abundance_matrix,
                diversity_features,
                interaction_features,
                network_features,
                pca_features
            ], axis=1)
            
            # 9. Scale features
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(all_features),
                columns=all_features.columns,
                index=all_features.index
            )
            
            # Save feature importance
            feature_importance = pd.Series(
                np.abs(self.pca.components_[0]),
                index=abundance_matrix.columns
            ).sort_values(ascending=False)
            
            feature_importance.to_csv(self.data_dir / "processed/feature_importance.csv")
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error in process_abundance_data: {str(e)}")
            raise
    
    def load_and_preprocess_data(self):
        """Load and preprocess data from multiple sources."""
        try:
            # Load processed abundance data
            abundance_train = pd.read_csv(self.data_dir / "processed/abundance_train.csv", index_col=0)
            
            # Load targets
            targets = pd.read_csv(self.data_dir / "processed/targets.csv", index_col=0)
            
            # Process abundance data
            processed_features = self.process_abundance_data(abundance_train)
            
            # Align samples
            common_samples = processed_features.index.intersection(targets.index)
            X = processed_features.loc[common_samples]
            y = targets.loc[common_samples]
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in load_and_preprocess_data: {str(e)}")
            raise 