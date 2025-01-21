import numpy as np
import pandas as pd
from scipy.stats import entropy
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.composition import clr, multiplicative_replacement
from networkx import Graph, pagerank
from typing import Dict, List, Tuple

class MicrobiomeFeatureEngineer:
    def __init__(
        self,
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        n_clusters: int = 20
    ):
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.n_clusters = n_clusters
        self.feature_names_ = []
        
    def _calculate_diversity_metrics(self, X: np.ndarray) -> np.ndarray:
        """Calculate comprehensive diversity metrics"""
        # Replace zeros with small value for log calculations
        X_nonzero = multiplicative_replacement(X)
        
        # Alpha diversity metrics
        shannon = alpha_diversity('shannon', X_nonzero)
        simpson = alpha_diversity('simpson', X_nonzero)
        chao1 = alpha_diversity('chao1', X_nonzero)
        observed_otus = alpha_diversity('observed_otus', X_nonzero)
        
        # Beta diversity metrics
        weighted_unifrac = beta_diversity('weighted_unifrac', X_nonzero)
        unweighted_unifrac = beta_diversity('unweighted_unifrac', X_nonzero)
        
        # Evenness and dominance
        evenness = shannon / np.log(observed_otus)
        berger_parker = np.max(X_nonzero, axis=1)
        
        return np.column_stack([
            shannon, simpson, chao1, observed_otus,
            weighted_unifrac.data, unweighted_unifrac.data,
            evenness, berger_parker
        ])
    
    def _calculate_ratios(self, X: np.ndarray) -> np.ndarray:
        """Calculate important taxonomic ratios"""
        # Example ratios (customize based on domain knowledge)
        firmicutes_idx = self.feature_names_.index('Firmicutes')
        bacteroidetes_idx = self.feature_names_.index('Bacteroidetes')
        
        f_b_ratio = X[:, firmicutes_idx] / (X[:, bacteroidetes_idx] + 1e-10)
        
        # Add more relevant ratios based on literature
        return f_b_ratio.reshape(-1, 1)
    
    def _create_co_occurrence_network(self, X: np.ndarray) -> np.ndarray:
        """Create and analyze microbial co-occurrence network"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Create network
        G = Graph()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > 0.3:  # Correlation threshold
                    G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
        
        # Calculate network metrics
        pagerank_scores = pagerank(G)
        centrality = np.zeros(X.shape[1])
        for node, score in pagerank_scores.items():
            centrality[node] = score
        
        return centrality.reshape(-1, 1)
    
    def _calculate_metabolic_potential(self, X: np.ndarray) -> np.ndarray:
        """Estimate metabolic potential using taxonomic abundances"""
        # This is a simplified example - expand based on known metabolic pathways
        metabolic_scores = np.zeros((X.shape[0], 3))
        
        # Example metabolic pathways (customize based on domain knowledge)
        carb_metabolism = np.sum(X[:, [1, 3, 5]], axis=1)  # Example indices
        protein_metabolism = np.sum(X[:, [2, 4, 6]], axis=1)
        lipid_metabolism = np.sum(X[:, [0, 7, 8]], axis=1)
        
        metabolic_scores[:, 0] = carb_metabolism
        metabolic_scores[:, 1] = protein_metabolism
        metabolic_scores[:, 2] = lipid_metabolism
        
        return metabolic_scores
    
    def fit_transform(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """Generate comprehensive microbiome features"""
        self.feature_names_ = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]
        
        # Filter low prevalence/abundance features
        prevalence = np.sum(X > 0, axis=0) / X.shape[0]
        abundance = np.mean(X, axis=0)
        keep_features = (prevalence >= self.min_prevalence) & (abundance >= self.min_abundance)
        X_filtered = X[:, keep_features]
        
        # Calculate base features
        diversity_features = self._calculate_diversity_metrics(X_filtered)
        ratio_features = self._calculate_ratios(X_filtered)
        network_features = self._create_co_occurrence_network(X_filtered)
        metabolic_features = self._calculate_metabolic_potential(X_filtered)
        
        # CLR transformation for compositional data
        X_clr = clr(multiplicative_replacement(X_filtered))
        
        # Combine all features
        X_combined = np.hstack([
            X_clr,
            diversity_features,
            ratio_features,
            network_features,
            metabolic_features
        ])
        
        return X_combined 