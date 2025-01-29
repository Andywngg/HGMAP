# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class MicrobiomeProcessor:
    def __init__(
        self,
        data_dir: str = "data",
        min_prevalence: float = 0.1,
        min_abundance: float = 0.001,
        random_state: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.min_prevalence = min_prevalence
        self.min_abundance = min_abundance
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.pca = PCA(n_components=50, random_state=random_state)
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=200)
        self.logger = logging.getLogger(__name__)
        
    def compute_diversity_metrics(self, abundance_matrix):
        """Compute various diversity metrics."""
        # Convert to proportions
        proportions = abundance_matrix.div(abundance_matrix.sum(axis=1), axis=0)
        
        # Shannon diversity
        shannon = -(proportions * np.log1p(proportions)).sum(axis=1)
        
        # Simpson diversity
        simpson = 1 - (proportions ** 2).sum(axis=1)
        
        # Species richness
        richness = (abundance_matrix > 0).sum(axis=1)
        
        # Evenness
        evenness = shannon / np.log1p(richness)
        
        # Dominance
        dominance = abundance_matrix.max(axis=1) / abundance_matrix.sum(axis=1)
        
        return pd.DataFrame({
            'shannon_diversity': shannon,
            'simpson_diversity': simpson,
            'species_richness': richness,
            'evenness': evenness,
            'dominance': dominance
        })
    
    def compute_interaction_features(self, abundance_matrix):
        """Compute microbial interaction features."""
        # Get top abundant species
        mean_abundance = abundance_matrix.mean()
        top_species = mean_abundance.nlargest(20).index
        
        interaction_features = {}
        for i, sp1 in enumerate(top_species):
            for sp2 in top_species[i+1:]:
                col_name = f"interaction_{sp1}_{sp2}"
                interaction_features[col_name] = abundance_matrix[sp1] * abundance_matrix[sp2]
        
        return pd.DataFrame(interaction_features)
    
    def compute_network_features(self, abundance_matrix):
        """Compute network-based features."""
        correlation_matrix = abundance_matrix.corr()
        
        return pd.DataFrame({
            'mean_correlation': correlation_matrix.mean(),
            'max_correlation': correlation_matrix.max(),
            'min_correlation': correlation_matrix.min(),
            'correlation_std': correlation_matrix.std()
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
            
            # 3. Log transformation
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
            
            return scaled_features
            
        except Exception as e:
            self.logger.error(f"Error in process_abundance_data: {str(e)}")
            raise
    
    def integrate_datasets(self):
        """Integrate data from multiple sources."""
        try:
            datasets = []
            
            # Load MGnify data
            mgnify_path = self.data_dir / "mgnify/abundance_final.csv"
            if mgnify_path.exists():
                mgnify_df = pd.read_csv(mgnify_path, index_col=0)
                datasets.append(('mgnify', mgnify_df))
            
            # Load AGP data
            agp_path = self.data_dir / "american_gut/abundance_final.csv"
            if agp_path.exists():
                agp_df = pd.read_csv(agp_path, index_col=0)
                datasets.append(('agp', agp_df))
            
            # Load HMP data
            hmp_path = self.data_dir / "hmp/abundance_final.csv"
            if hmp_path.exists():
                hmp_df = pd.read_csv(hmp_path, index_col=0)
                datasets.append(('hmp', hmp_df))
            
            if not datasets:
                raise ValueError("No datasets found")
            
            # Align features across datasets
            common_features = set.intersection(*[set(df.columns) for _, df in datasets])
            
            # Combine datasets
            combined_data = pd.concat([
                df[list(common_features)] for _, df in datasets
            ], axis=0)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error in integrate_datasets: {str(e)}")
            raise
    
    def prepare_data(self):
        """Main method to prepare data for modeling."""
        try:
            # 1. Integrate datasets
            self.logger.info("Integrating datasets...")
            abundance_data = self.integrate_datasets()
            
            # 2. Process abundance data
            self.logger.info("Processing abundance data...")
            processed_data = self.process_abundance_data(abundance_data)
            
            # 3. Load health status labels
            self.logger.info("Loading health status labels...")
            labels = []
            for source in ['mgnify', 'agp', 'hmp']:
                label_path = self.data_dir / f"{source}/metadata_final.csv"
                if label_path.exists():
                    df = pd.read_csv(label_path)
                    labels.append(df)
            
            if not labels:
                raise ValueError("No label data found")
            
            labels_df = pd.concat(labels, axis=0)
            
            # 4. Align samples
            common_samples = set(processed_data.index) & set(labels_df['sample_id'])
            if not common_samples:
                raise ValueError("No common samples between features and labels")
            
            X = processed_data.loc[list(common_samples)]
            y = labels_df[labels_df['sample_id'].isin(common_samples)]['health_status'].map({'Healthy': 0, 'Non-healthy': 1})
            
            self.logger.info(f"Final dataset shape: {X.shape}")
            self.logger.info(f"Class distribution: {y.value_counts()}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def train_and_evaluate(self):
        try:
            X, y = self.prepare_data()
            
            # Create the full pipeline with multiple models
            models = {
                "rf": RandomForestClassifier(
                    n_estimators=2000,
                    max_depth=20,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42
                ),
                "gb": GradientBoostingClassifier(
                    n_estimators=500,
                    learning_rate=0.1,
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                )
            }
            
            best_score = 0
            best_model = None
            
            for name, model in models.items():
                pipeline = Pipeline([
                    ("feature_prep", self.prepare_data()),
                    ("classifier", model)
                ])
                
                # Evaluate using stratified k-fold
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="balanced_accuracy")
                
                self.logger.info(f"{name.upper()} - Cross-validation scores: {scores}")
                self.logger.info(f"{name.upper()} - Mean balanced accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
                
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_model = pipeline
            
            # Fine-tune the best model
            if isinstance(best_model.named_steps["classifier"], RandomForestClassifier):
                param_grid = {
                    "classifier__n_estimators": [1500, 2000, 2500],
                    "classifier__max_depth": [15, 20, 25],
                    "feature_prep__selector__k": [150, 200, 250]
                }
            else:
                param_grid = {
                    "classifier__n_estimators": [400, 500, 600],
                    "classifier__learning_rate": [0.05, 0.1, 0.15],
                    "feature_prep__selector__k": [150, 200, 250]
                }
            
            grid_search = GridSearchCV(
                best_model,
                param_grid,
                cv=5,
                scoring="balanced_accuracy",
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            # Save results
            results = {
                "best_params": grid_search.best_params_,
                "best_score": float(grid_search.best_score_),
                "cv_results": grid_search.cv_results_
            }
            
            return grid_search.best_estimator_, results
            
        except Exception as e:
            self.logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

if __name__ == "__main__":
    processor = MicrobiomeProcessor()
    processor.train_and_evaluate()