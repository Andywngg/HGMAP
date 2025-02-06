#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for generating visualizations of data and model results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import shap

class VisualizationManager:
    """Class for managing and generating visualizations."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        feature_importance_config: Dict,
        shap_config: Dict
    ):
        """
        Initialize the VisualizationManager.
        
        Args:
            output_dir: Directory to save visualizations
            feature_importance_config: Configuration for feature importance plots
            shap_config: Configuration for SHAP plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_importance_config = feature_importance_config
        self.shap_config = shap_config
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_feature_distributions(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        n_cols: int = 3,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        Plot distribution of features, colored by target class.
        
        Args:
            features_df: DataFrame containing features
            target: Series containing target labels
            n_cols: Number of columns in the subplot grid
            figsize: Figure size for each subplot
        """
        n_features = features_df.shape[1]
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        axes = axes.ravel()
        
        for idx, column in enumerate(features_df.columns):
            sns.kdeplot(
                data=features_df,
                x=column,
                hue=target,
                ax=axes[idx],
                common_norm=False
            )
            axes[idx].set_title(f'Distribution of {column}')
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png')
        plt.close()
    
    def plot_correlation_matrix(
        self,
        features_df: pd.DataFrame,
        method: str = 'spearman',
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            features_df: DataFrame containing features
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            figsize: Figure size
        """
        corr_matrix = features_df.corr(method=method)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'correlation_matrix_{method}.png')
        plt.close()
    
    def plot_pca_components(
        self,
        features: np.ndarray,
        target: np.ndarray,
        n_components: int = 2,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot PCA components with target classes.
        
        Args:
            features: Array of features
            target: Array of target labels
            n_components: Number of components to plot
            figsize: Figure size
        """
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(features)
        
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            components[:, 0],
            components[:, 1],
            c=target,
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Components')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_components.png')
        plt.close()
        
        # Plot explained variance ratio
        plt.figure(figsize=(8, 4))
        plt.plot(
            range(1, n_components + 1),
            np.cumsum(pca.explained_variance_ratio_),
            'bo-'
        )
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_explained_variance.png')
        plt.close()
    
    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str],
        title: str = 'Feature Importance'
    ) -> None:
        """
        Plot feature importance scores.
        
        Args:
            importance: Array of feature importance scores
            feature_names: List of feature names
            title: Plot title
        """
        n_features = min(
            len(importance),
            self.feature_importance_config['n_top_features']
        )
        
        # Sort features by importance
        indices = np.argsort(importance)[-n_features:]
        
        plt.figure(figsize=tuple(self.feature_importance_config['figsize']))
        plt.barh(range(n_features), importance[indices])
        plt.yticks(range(n_features), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
    
    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str]
    ) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values: SHAP values for features
            features: Feature matrix
            feature_names: List of feature names
        """
        plt.figure(figsize=tuple(self.shap_config['figsize']))
        shap.summary_plot(
            shap_values,
            features,
            feature_names=feature_names,
            max_display=self.shap_config['max_display'],
            plot_type=self.shap_config['plot_type'],
            show=False
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shap_summary.png')
        plt.close()
    
    def plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        features: np.ndarray,
        feature_names: List[str],
        feature_idx: int
    ) -> None:
        """
        Plot SHAP dependence plot for a specific feature.
        
        Args:
            shap_values: SHAP values for features
            features: Feature matrix
            feature_names: List of feature names
            feature_idx: Index of feature to plot
        """
        plt.figure(figsize=tuple(self.shap_config['figsize']))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            features,
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'shap_dependence_{feature_names[feature_idx]}.png'
        )
        plt.close()
    
    def plot_learning_curves(
        self,
        train_scores: List[float],
        val_scores: List[float],
        metric: str = 'loss',
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Plot learning curves showing training and validation scores.
        
        Args:
            train_scores: List of training scores
            val_scores: List of validation scores
            metric: Name of the metric being plotted
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, 'b-', label='Training')
        plt.plot(epochs, val_scores, 'r-', label='Validation')
        
        plt.title(f'Learning Curves - {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'learning_curves_{metric}.png')
        plt.close()
    
    def plot_class_distribution(
        self,
        target: np.ndarray,
        title: str = 'Class Distribution',
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        Plot distribution of target classes.
        
        Args:
            target: Array of target labels
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        sns.countplot(x=target)
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png')
        plt.close() 