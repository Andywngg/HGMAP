#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.decomposition import PCA
import logging
from typing import Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MicrobiomeFeatureEngineer:
    """Engineer features from microbiome abundance data."""
    
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.feature_names = []
        
    def calculate_alpha_diversity(self, abundance_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate alpha diversity metrics."""
        # Normalize abundances
        rel_abundance = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
        
        # Shannon diversity
        shannon = rel_abundance.apply(lambda x: entropy(x[x > 0]), axis=1)
        
        # Species richness (number of non-zero species)
        richness = (abundance_matrix > 0).sum(axis=1)
        
        # Pielou's evenness (Shannon diversity / log(richness))
        evenness = shannon / np.log(richness)
        
        # Simpson's diversity
        simpson = rel_abundance.apply(lambda x: 1 - np.sum(x**2), axis=1)
        
        diversity_df = pd.DataFrame({
            'shannon_diversity': shannon,
            'richness': richness,
            'evenness': evenness,
            'simpson_diversity': simpson
        })
        
        return diversity_df
    
    def calculate_phylum_ratios(self, abundance_matrix: pd.DataFrame, taxonomy: pd.Series) -> pd.DataFrame:
        """Calculate important phylum ratios."""
        # Extract phylum from taxonomy string
        def get_phylum(tax_string):
            phyla = [t.strip() for t in tax_string.split(';') if 'p__' in t]
            return phyla[0].replace('p__', '') if phyla else 'Unknown'
        
        phylum_abundance = pd.DataFrame()
        for idx, row in abundance_matrix.iterrows():
            phylum = get_phylum(taxonomy[idx])
            if phylum not in phylum_abundance.columns:
                phylum_abundance[phylum] = 0
            phylum_abundance.loc[idx, phylum] = row.sum()
        
        # Calculate important ratios
        ratios = pd.DataFrame()
        
        # Firmicutes to Bacteroidetes ratio
        if 'Firmicutes' in phylum_abundance.columns and 'Bacteroidetes' in phylum_abundance.columns:
            ratios['firmicutes_bacteroidetes_ratio'] = (
                phylum_abundance['Firmicutes'] / 
                phylum_abundance['Bacteroidetes'].replace(0, np.nan)
            )
        
        # Proteobacteria abundance
        if 'Proteobacteria' in phylum_abundance.columns:
            ratios['proteobacteria_abundance'] = phylum_abundance['Proteobacteria']
        
        return ratios
    
    def calculate_beta_diversity(self, abundance_matrix: pd.DataFrame) -> pd.DataFrame:
        """Calculate beta diversity using PCA."""
        # Normalize abundances
        rel_abundance = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(rel_abundance)
        
        # Create DataFrame with PCA features
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'PCA_{i+1}' for i in range(self.n_components)]
        )
        
        # Store explained variance ratio
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return pca_df
    
    def engineer_features(self, abundance_matrix: pd.DataFrame, taxonomy: pd.Series) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        logger.info("Calculating alpha diversity metrics...")
        alpha_diversity = self.calculate_alpha_diversity(abundance_matrix)
        
        logger.info("Calculating phylum ratios...")
        phylum_ratios = self.calculate_phylum_ratios(abundance_matrix, taxonomy)
        
        logger.info("Calculating beta diversity metrics...")
        beta_diversity = self.calculate_beta_diversity(abundance_matrix)
        
        # Combine all features
        features = pd.concat([
            alpha_diversity,
            phylum_ratios,
            beta_diversity
        ], axis=1)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        logger.info(f"Engineered {len(self.feature_names)} features")
        return features
    
    def save_features(self, features: pd.DataFrame, output_dir: str = 'features'):
        """Save engineered features and metadata."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features.to_csv(output_dir / 'engineered_features.csv')
        
        # Save feature metadata
        feature_metadata = pd.DataFrame({
            'feature_name': self.feature_names,
            'feature_type': [
                'alpha_diversity' if 'diversity' in f or f in ['richness', 'evenness']
                else 'phylum_ratio' if 'ratio' in f or 'abundance' in f
                else 'beta_diversity' if 'PCA' in f
                else 'other'
                for f in self.feature_names
            ]
        })
        
        if hasattr(self, 'explained_variance_ratio'):
            pca_importance = pd.DataFrame({
                'component': [f'PCA_{i+1}' for i in range(len(self.explained_variance_ratio))],
                'explained_variance_ratio': self.explained_variance_ratio,
                'cumulative_variance_ratio': np.cumsum(self.explained_variance_ratio)
            })
            pca_importance.to_csv(output_dir / 'pca_importance.csv', index=False)
        
        feature_metadata.to_csv(output_dir / 'feature_metadata.csv', index=False) 