import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from xgboost import XGBClassifier
import warnings
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnhancedMicrobiomeProcessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.pca = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        
        # Initialize multiple models
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            ),
            'xgb': XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state
            )
        }
        
        self.best_model = None
        
    def _engineer_features(self, X):
        """Create advanced features"""
        features = X.copy()
        
        # Handle NaN values in the input features
        features = features.fillna(features.mean())
        
        # Create interaction terms
        numeric_cols = features.columns
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                features[f'interaction_{col1}_{col2}'] = features[col1] * features[col2]
        
        # Add polynomial features for diversity metrics
        diversity_metrics = ['shannon_diversity', 'simpson_diversity', 'species_richness']
        for col in diversity_metrics:
            if col in features.columns:
                features[f'{col}_squared'] = features[col] ** 2
                features[f'{col}_cubed'] = features[col] ** 3
        
        # Add ratio features (with error handling)
        features['gmwi_ratio'] = features['gmwi'] / (features['gmwi2'] + 1e-10)  # Add small constant to prevent division by zero
        features['diversity_ratio'] = features['shannon_diversity'] / (features['simpson_diversity'] + 1e-10)
        features['richness_normalized'] = features['species_richness'] / (features['species_richness'].max() + 1e-10)
        
        # Add exponential and log transformations (with error handling)
        for col in numeric_cols:
            # Normalize to [0,1] range for exp to prevent overflow
            normalized = (features[col] - features[col].min()) / (features[col].max() - features[col].min() + 1e-10)
            features[f'{col}_exp'] = np.exp(normalized)
            # Add small constant for log to handle zeros
            features[f'{col}_log'] = np.log1p(features[col] - features[col].min() + 1e-10)
        
        # Handle any NaN values that might have been created
        features = features.fillna(0)
        
        # Replace infinite values with large finite numbers
        features = features.replace([np.inf, -np.inf], [1e10, -1e10])
        
        return features
        
    def _optimize_model(self, X, y, model_type='xgb'):
        """Optimize model hyperparameters"""
        if model_type == 'xgb':
            # Convert data to DMatrix format
            dtrain = xgb.DMatrix(X, label=y)
            
            # Set parameters
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': self.random_state,
                'nthread': -1
            }
            
            # Train model
            num_round = 500
            model = xgb.train(params, dtrain, num_round)
            
            # Create sklearn-compatible wrapper
            sklearn_model = XGBClassifier(**params, n_estimators=num_round)
            sklearn_model.fit(X, y)
            
            # Get cross-validation score
            cv_scores = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_round,
                nfold=5,
                metrics='error',
                seed=self.random_state
            )
            
            final_score = 1 - cv_scores['test-error-mean'].iloc[-1]
            final_std = cv_scores['test-error-std'].iloc[-1]
            self.logger.info(f"Cross-validation accuracy: {final_score:.3f} (+/- {final_std * 2:.3f})")
            
            return sklearn_model
            
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
            
        if model_type in ['rf', 'gb']:
            # Fit model
            model.fit(X, y)
            
            # Get cross-validation score
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            self.logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return model
        
    def process_integrated_data(self, abundance_path, metadata_path):
        try:
            # Load and preprocess data
            X, y = self._load_and_preprocess(abundance_path, metadata_path)
            
            # Feature engineering and selection
            X_selected = self._engineer_and_select_features(X)
            
            # Compute class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y),
                y=y
            )
            weight_dict = dict(zip(np.unique(y), class_weights))
            sample_weights = np.array([weight_dict[label] for label in y])
            
            # Select features using mutual information
            k = min(50, X_selected.shape[1])  # Select top 50 features or all if less
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected_mi = selector.fit_transform(X_selected, y)
            selected_features = X_selected.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected_mi, columns=selected_features)
            
            # Initialize XGBoost with optimized parameters
            params = {
                'max_depth': 8,
                'learning_rate': 0.03,
                'subsample': 0.85,
                'colsample_bytree': 0.75,
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'error', 'auc'],
                'seed': 42,
                'num_boost_round': 2000,
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'max_leaves': 128,
                'min_child_weight': 3,
                'gamma': 0.3,
                'reg_alpha': 0.3,
                'reg_lambda': 1.5,
                'scale_pos_weight': 1.0,
                'max_bin': 256,
                'early_stopping_rounds': 100
            }
            
            self.best_model = XGBoostWrapper(**params)
            
            # Perform cross-validation with stratification and sample weights
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X_selected, y):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                w_train = sample_weights[train_idx]
                
                # Create a new model instance for each fold
                model = XGBoostWrapper(**params)
                
                # Train on this fold
                dtrain = xgb.DMatrix(X_train.values, label=y_train, weight=w_train)
                dval = xgb.DMatrix(X_val.values, label=y_val)
                evallist = [(dtrain, 'train'), (dval, 'eval')]
                
                model.model = xgb.train(
                    model.params,
                    dtrain,
                    num_boost_round=model.num_boost_round,
                    evals=evallist,
                    early_stopping_rounds=model.early_stopping_rounds,
                    verbose_eval=False
                )
                
                # Evaluate on validation set
                scores.append(model.score(X_val.values, y_val))
            
            self.logger.info(f"Cross-validation accuracy: {np.mean(scores):.3f} (+/- {np.std(scores) * 2:.3f})")
            
            # Train final model on full dataset
            dtrain_full = xgb.DMatrix(X_selected.values, label=y, weight=sample_weights)
            self.best_model.model = xgb.train(
                self.best_model.params,
                dtrain_full,
                num_boost_round=self.best_model.num_boost_round
            )
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X_selected.columns,
                'importance': np.zeros(len(X_selected.columns))
            })
            
            try:
                importance_dict = self.best_model.model.get_score(importance_type='gain')
                for feature_id, importance in importance_dict.items():
                    idx = int(feature_id.replace('f', ''))
                    if idx < len(feature_importance):
                        feature_importance.loc[idx, 'importance'] = importance
            except Exception as e:
                self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            return {
                'model': self.best_model,
                'features': X_selected.columns.tolist(),
                'cv_scores': scores,
                'cv_mean': float(np.mean(scores)),
                'cv_std': float(np.std(scores)),
                'feature_importance': feature_importance.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Error processing integrated data: {str(e)}")
            raise
            
    def _get_feature_importance(self, X):
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            selected_features = self.feature_selector.get_feature_names_out(self.feature_names)
            return dict(zip(selected_features, self.best_model.feature_importances_))
        return None
        
    def generate_report(self, results):
        """Generate a comprehensive analysis report"""
        report = {
            'data_summary': {
                'n_features': len(results['features']),
                'features': results['features']
            },
            'model_performance': {
                'mean_cv_accuracy': results['cv_mean'],
                'std_cv_accuracy': results['cv_std'],
                'individual_scores': results['cv_scores']
            }
        }
        
        if results['feature_importance']:
            # Get top 10 important features
            top_features = sorted(
                results['feature_importance'], 
                key=lambda x: x['importance'], 
                reverse=True
            )[:10]
            report['top_features'] = top_features
            
        return report 

    def _load_and_preprocess(self, abundance_path, metadata_path):
        """Load and preprocess the data."""
        try:
            # Load abundance data
            self.logger.info("Loading abundance data...")
            abundance_data = pd.read_csv(abundance_path)
            
            # Load metadata
            self.logger.info("Loading metadata...")
            metadata = pd.read_csv(metadata_path)
            
            # Merge data
            self.logger.info("Merging data...")
            merged_data = pd.merge(
                abundance_data,
                metadata[['sample_id', 'health_status']],
                on='sample_id',
                how='inner'
            )
            
            if merged_data.empty:
                raise ValueError("No data after merging")
            
            # Prepare features and labels
            X = merged_data.drop(['sample_id', 'health_status'], axis=1)
            y = self.label_encoder.fit_transform(merged_data['health_status'])
            
            self.feature_names = X.columns.tolist()
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in _load_and_preprocess: {str(e)}")
            raise

    def _engineer_and_select_features(self, X):
        """Engineer and select features."""
        try:
            # Engineer features
            self.logger.info("Engineering features...")
            
            # Create a copy of the dataframe
            X_engineered = X.copy()
            
            # Handle missing values using forward fill then backward fill
            X_engineered = X_engineered.fillna(method='ffill').fillna(method='bfill')
            
            # Create polynomial features for all numeric columns
            numeric_cols = X_engineered.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                X_engineered[f'{col}_squared'] = X_engineered[col] ** 2
                X_engineered[f'{col}_cubed'] = X_engineered[col] ** 3
                # Add log transformation (with offset to handle zeros/negatives)
                min_val = X_engineered[col].min()
                offset = abs(min_val) + 1 if min_val <= 0 else 0
                X_engineered[f'{col}_log'] = np.log1p(X_engineered[col] + offset)
            
            # Create interaction terms for all numeric columns
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    # Multiplication interaction
                    X_engineered[f'interaction_mult_{col1}_{col2}'] = X_engineered[col1] * X_engineered[col2]
                    # Division interaction (with safety)
                    X_engineered[f'interaction_div_{col1}_{col2}'] = X_engineered[col1] / (X_engineered[col2] + 1e-10)
                    # Addition interaction
                    X_engineered[f'interaction_add_{col1}_{col2}'] = X_engineered[col1] + X_engineered[col2]
                    # Subtraction interaction
                    X_engineered[f'interaction_sub_{col1}_{col2}'] = X_engineered[col1] - X_engineered[col2]
            
            # Create domain-specific features
            if 'shannon_diversity' in X_engineered.columns and 'simpson_diversity' in X_engineered.columns:
                # Diversity ratio
                X_engineered['diversity_ratio'] = X_engineered['shannon_diversity'] / (X_engineered['simpson_diversity'] + 1e-10)
                # Diversity product
                X_engineered['diversity_product'] = X_engineered['shannon_diversity'] * X_engineered['simpson_diversity']
                # Diversity difference
                X_engineered['diversity_diff'] = X_engineered['shannon_diversity'] - X_engineered['simpson_diversity']
                
            if 'species_richness' in X_engineered.columns:
                # Normalize richness
                X_engineered['richness_normalized'] = X_engineered['species_richness'] / (X_engineered['species_richness'].max() + 1e-10)
                # Log richness
                X_engineered['richness_log'] = np.log1p(X_engineered['species_richness'])
                
            if 'gmwi' in X_engineered.columns and 'gmwi2' in X_engineered.columns:
                # GMWI features
                X_engineered['gmwi_ratio'] = X_engineered['gmwi'] / (X_engineered['gmwi2'] + 1e-10)
                X_engineered['gmwi_sum'] = X_engineered['gmwi'] + X_engineered['gmwi2']
                X_engineered['gmwi_diff'] = X_engineered['gmwi'] - X_engineered['gmwi2']
                X_engineered['gmwi_product'] = X_engineered['gmwi'] * X_engineered['gmwi2']
            
            # Remove constant and highly correlated features
            self.logger.info("Removing constant and highly correlated features...")
            
            # Remove constant features
            constant_filter = (X_engineered.std() != 0)
            X_engineered = X_engineered.loc[:, constant_filter]
            
            # Remove highly correlated features using a lower threshold
            corr_matrix = X_engineered.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]  # Lowered threshold to 0.90
            X_engineered = X_engineered.drop(to_drop, axis=1)
            
            self.logger.info(f"Features after correlation removal: {X_engineered.shape[1]}")
            
            # Scale features
            self.logger.info("Scaling features...")
            X_scaled = self.scaler.fit_transform(X_engineered)
            X_scaled = pd.DataFrame(X_scaled, columns=X_engineered.columns)
            
            # Handle any remaining NaN or infinite values
            X_scaled = X_scaled.replace([np.inf, -np.inf], [1e10, -1e10])
            X_scaled = X_scaled.fillna(0)
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"Error in _engineer_and_select_features: {str(e)}")
            raise

class XGBoostWrapper(BaseEstimator):
    def __init__(self, **params):
        self.params = {
            'max_depth': params.get('max_depth', 8),
            'learning_rate': params.get('learning_rate', 0.03),
            'subsample': params.get('subsample', 0.85),
            'colsample_bytree': params.get('colsample_bytree', 0.75),
            'objective': 'binary:logistic',
            'eval_metric': params.get('eval_metric', ['logloss', 'error', 'auc']),
            'seed': params.get('seed', 42),
            'tree_method': params.get('tree_method', 'hist'),
            'grow_policy': params.get('grow_policy', 'lossguide'),
            'max_leaves': params.get('max_leaves', 128),
            'min_child_weight': params.get('min_child_weight', 3),
            'gamma': params.get('gamma', 0.3),
            'reg_alpha': params.get('reg_alpha', 0.3),
            'reg_lambda': params.get('reg_lambda', 1.5),
            'scale_pos_weight': params.get('scale_pos_weight', 1.0),
            'max_bin': params.get('max_bin', 256)
        }
        self.num_boost_round = params.get('num_boost_round', 2000)
        self.early_stopping_rounds = params.get('early_stopping_rounds', 100)
        self.model = None
        self._estimator_type = "classifier"
        
    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
        else:
            dtrain = xgb.DMatrix(X, label=y)
            
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round
        )
        return self
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return (self.model.predict(dtest) > 0.5).astype(int)
        
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.model.predict(dtest)
        return np.vstack([1-preds, preds]).T
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
        
    def get_params(self, deep=True):
        params = self.params.copy()
        params['num_boost_round'] = self.num_boost_round
        params['early_stopping_rounds'] = self.early_stopping_rounds
        return params
        
    def set_params(self, **params):
        for param, value in params.items():
            if param == 'num_boost_round':
                self.num_boost_round = value
            elif param == 'early_stopping_rounds':
                self.early_stopping_rounds = value
            else:
                self.params[param] = value
        return self 