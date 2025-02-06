import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
import logging

logger = logging.getLogger(__name__)

class DiversityCalculator:
    """Calculate various alpha and beta diversity metrics for microbiome data."""
    
    def __init__(self):
        pass
    
    def calculate_alpha_diversity(self, abundance_matrix: np.ndarray) -> pd.DataFrame:
        """Calculate alpha diversity metrics."""
        metrics = []
        
        for sample_abundances in abundance_matrix:
            # Filter out zeros for calculations
            nonzero_abundances = sample_abundances[sample_abundances > 0]
            total_abundance = np.sum(nonzero_abundances)
            
            # Normalize abundances
            if total_abundance > 0:
                normalized_abundances = nonzero_abundances / total_abundance
            else:
                normalized_abundances = nonzero_abundances
            
            # Calculate metrics
            sample_metrics = {
                # Richness (number of observed species)
                'richness': len(nonzero_abundances),
                
                # Shannon diversity
                'shannon_diversity': entropy(normalized_abundances) if len(normalized_abundances) > 0 else 0,
                
                # Simpson diversity
                'simpson_diversity': 1 - np.sum(normalized_abundances ** 2) if len(normalized_abundances) > 0 else 0,
                
                # Pielou's evenness
                'pielou_evenness': (entropy(normalized_abundances) / np.log(len(normalized_abundances))) 
                                  if len(normalized_abundances) > 1 else 0,
                
                # Chao1 richness estimator
                'chao1': self._calculate_chao1(sample_abundances),
                
                # ACE (Abundance-based Coverage Estimator)
                'ace': self._calculate_ace(sample_abundances),
                
                # Fisher's alpha
                'fisher_alpha': self._calculate_fisher_alpha(sample_abundances)
            }
            metrics.append(sample_metrics)
        
        return pd.DataFrame(metrics)
    
    def calculate_beta_diversity(
        self,
        abundance_matrix: np.ndarray,
        metric: str = 'bray-curtis'
    ) -> np.ndarray:
        """Calculate beta diversity metrics."""
        # Normalize abundance matrix
        row_sums = abundance_matrix.sum(axis=1)
        normalized_matrix = abundance_matrix / row_sums[:, np.newaxis]
        
        if metric == 'bray-curtis':
            distances = pairwise_distances(normalized_matrix, metric='braycurtis')
        elif metric == 'jaccard':
            distances = pairwise_distances(normalized_matrix > 0, metric='jaccard')
        elif metric == 'unifrac':
            # Note: UniFrac would require phylogenetic tree information
            logger.warning("UniFrac calculation requires phylogenetic tree information")
            distances = pairwise_distances(normalized_matrix, metric='euclidean')
        else:
            raise ValueError(f"Unsupported beta diversity metric: {metric}")
        
        return distances
    
    def _calculate_chao1(self, abundances: np.ndarray) -> float:
        """Calculate Chao1 richness estimator."""
        # Count singletons and doubletons
        unique, counts = np.unique(abundances[abundances > 0], return_counts=True)
        singletons = np.sum(counts == 1)
        doubletons = np.sum(counts == 2)
        
        # Calculate Chao1
        if doubletons > 0:
            chao1 = len(unique) + (singletons * singletons) / (2 * doubletons)
        else:
            chao1 = len(unique) + singletons * (singletons - 1) / 2
        
        return chao1
    
    def _calculate_ace(self, abundances: np.ndarray, rare_threshold: int = 10) -> float:
        """Calculate ACE (Abundance-based Coverage Estimator)."""
        # Count species abundances
        unique, counts = np.unique(abundances[abundances > 0], return_counts=True)
        
        # Separate rare and abundant species
        rare_counts = counts[counts <= rare_threshold]
        n_rare = len(rare_counts)
        n_abundant = len(counts) - n_rare
        
        if n_rare == 0:
            return float(n_abundant)
        
        # Calculate components
        s_rare = np.sum(rare_counts)
        f1 = np.sum(rare_counts == 1)
        gamma2 = max(0, (n_rare * f1 / s_rare) * ((s_rare - 1) / s_rare))
        
        # Calculate ACE
        if s_rare > 0:
            c_ace = 1 - f1 / s_rare
            ace = n_abundant + n_rare / c_ace + (f1 / c_ace) * gamma2
        else:
            ace = n_abundant
        
        return ace
    
    def _calculate_fisher_alpha(self, abundances: np.ndarray, iterations: int = 100) -> float:
        """Calculate Fisher's alpha using iterative method."""
        # Count species abundances
        unique, counts = np.unique(abundances[abundances > 0], return_counts=True)
        n = np.sum(abundances)
        s = len(unique)
        
        # Initial guess for alpha
        alpha = s
        
        # Iterative solution
        for _ in range(iterations):
            alpha_new = s / (np.log(1 + n/alpha))
            if abs(alpha_new - alpha) < 0.001:
                break
            alpha = alpha_new
        
        return alpha
    
    def calculate_all_metrics(
        self,
        abundance_df: pd.DataFrame,
        beta_metrics: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """Calculate all diversity metrics."""
        if beta_metrics is None:
            beta_metrics = ['bray-curtis', 'jaccard']
        
        # Calculate alpha diversity
        alpha_div = self.calculate_alpha_diversity(abundance_df.values)
        
        # Calculate beta diversity for each metric
        beta_div = {}
        for metric in beta_metrics:
            beta_div[metric] = pd.DataFrame(
                self.calculate_beta_diversity(abundance_df.values, metric),
                index=abundance_df.index,
                columns=abundance_df.index
            )
        
        return {
            'alpha_diversity': alpha_div,
            'beta_diversity': beta_div
        } 