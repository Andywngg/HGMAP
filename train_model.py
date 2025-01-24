import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MicrobiomeModelTrainer:
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def engineer_features(self, features_df):
        """Create additional features from existing ones."""
        # Create interaction terms
        poly = PolynomialFeatures(degree=2, include_bias=False)
        feature_names = features_df.columns.tolist()
        interactions = poly.fit_transform(features_df)
        interaction_names = poly.get_feature_names_out(feature_names)
        
        # Create new features
        features_df['gmwi_ratio'] = features_df['gmwi2'] / features_df['gmwi']
        features_df['diversity_ratio'] = features_df['shannon_diversity'] / features_df['simpson_diversity']
        features_df['richness_shannon_ratio'] = features_df['species_richness'] / features_df['shannon_diversity']
        
        # Advanced diversity metrics
        features_df['diversity_product'] = features_df['shannon_diversity'] * features_df['simpson_diversity']
        features_df['normalized_richness'] = features_df['species_richness'] / features_df['species_richness'].max()
        features_df['gmwi_composite'] = (features_df['gmwi'] + features_df['gmwi2']) / 2
        
        # Statistical features
        features_df['diversity_zscore'] = (features_df['shannon_diversity'] - features_df['shannon_diversity'].mean()) / features_df['shannon_diversity'].std()
        features_df['richness_percentile'] = features_df['species_richness'].rank(pct=True)
        
        # Exponential and power transformations
        features_df['gmwi_exp'] = np.exp(features_df['gmwi'] / features_df['gmwi'].max())  # Normalized to prevent overflow
        features_df['diversity_squared'] = features_df['shannon_diversity'] ** 2
        
        # Log transform applicable features
        for col in features_df.columns:
            if features_df[col].min() > 0:  # Only log transform positive values
                features_df[f'{col}_log'] = np.log(features_df[col])
        
        # Interaction ratios
        features_df['gmwi_diversity_ratio'] = features_df['gmwi_composite'] / features_df['diversity_ratio']
        features_df['richness_gmwi_ratio'] = features_df['normalized_richness'] / features_df['gmwi_ratio']
        
        return features_df
        
    def load_data(self):
        """Load and preprocess the data."""
        try:
            # Load integrated data
            self.metadata = pd.read_csv("data/integrated/integrated_metadata.csv", index_col=0)
            self.features = pd.read_csv("data/integrated/abundance_data.csv", index_col=0)
            
            logging.info(f"Loaded data: {len(self.metadata)} samples")
            logging.info(f"Features shape: {self.features.shape}")
            
            # Select base features
            self.feature_names = ['gmwi2', 'gmwi', 'shannon_diversity', 'simpson_diversity', 'species_richness']
            features_df = self.features[self.feature_names].copy()
            
            # Engineer additional features
            features_df = self.engineer_features(features_df)
            self.feature_names = features_df.columns.tolist()
            
            # Advanced imputation using KNN
            imputer = KNNImputer(n_neighbors=5)
            X = imputer.fit_transform(features_df)
            y = (self.metadata['health_status'] == 'Non-healthy').astype(int)
            
            # Save feature names and imputer
            pd.DataFrame({'feature_name': self.feature_names}).to_csv(
                self.models_dir / "feature_names.csv", index=False
            )
            joblib.dump(imputer, self.models_dir / "imputer.joblib")
            
            logging.info(f"Processed features: {len(self.feature_names)}")
            logging.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
            
            return train_test_split(X, y, test_size=0.2, random_state=42)
        
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
    
    def train_models(self):
        """Train and save the models."""
        try:
            # Load and split data
            X_train, X_test, y_train, y_test = self.load_data()
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply SMOTE with more sophisticated sampling
            smote = SMOTE(random_state=42, k_neighbors=7)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            # Optimized parameter grids for faster training
            rf_params = {
                'n_estimators': [200],
                'max_depth': [15],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt']
            }
            
            gb_params = {
                'n_estimators': [200],
                'max_depth': [5],
                'learning_rate': [0.05],
                'subsample': [0.8],
                'min_samples_split': [5]
            }
            
            ada_params = {
                'n_estimators': [100],
                'learning_rate': [1.0],
                'algorithm': ['SAMME']
            }
            
            # Initialize and tune base models with parallel processing
            logging.info("Training Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_train_balanced, y_train_balanced)
            
            logging.info("Training Gradient Boosting...")
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                random_state=42
            )
            gb.fit(X_train_balanced, y_train_balanced)
            
            logging.info("Training AdaBoost...")
            ada = AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                algorithm='SAMME',
                random_state=42
            )
            ada.fit(X_train_balanced, y_train_balanced)
            
            # Create weighted voting ensemble
            estimators = [
                ('rf', rf),
                ('gb', gb),
                ('ada', ada)
            ]
            
            # Enhanced stacking with better meta-learner
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ),
                cv=5,
                n_jobs=-1
            )
            
            logging.info("Training Stacking Classifier...")
            stacking.fit(X_train_balanced, y_train_balanced)
            
            # Save models and scaler
            joblib.dump(rf, self.models_dir / "random_forest_model.joblib")
            joblib.dump(gb, self.models_dir / "gradient_boosting_model.joblib")
            joblib.dump(ada, self.models_dir / "adaboost_model.joblib")
            joblib.dump(stacking, self.models_dir / "stacking_model.joblib")
            joblib.dump(scaler, self.models_dir / "scaler.joblib")
            
            # Evaluate models with detailed metrics
            def evaluate_model(name, model, X_test, y_test):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                
                logging.info(f"\n{name} Performance:")
                logging.info(classification_report(y_test, y_pred))
                logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                
                # Calculate additional metrics
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)
                
                logging.info(f"Specificity: {specificity:.3f}")
                logging.info(f"Sensitivity: {sensitivity:.3f}")
                
                return y_pred, y_prob
            
            # Evaluate all models
            rf_pred, rf_prob = evaluate_model("Random Forest", rf, X_test_scaled, y_test)
            gb_pred, gb_prob = evaluate_model("Gradient Boosting", gb, X_test_scaled, y_test)
            ada_pred, ada_prob = evaluate_model("AdaBoost", ada, X_test_scaled, y_test)
            stack_pred, stack_prob = evaluate_model("Stacking Classifier", stacking, X_test_scaled, y_test)
            
            # Feature importance analysis with confidence intervals
            n_iterations = 50  # Reduced from 100 for faster processing
            importances = np.zeros((n_iterations, len(self.feature_names)))
            
            for i in range(n_iterations):
                # Bootstrap the training data
                indices = np.random.choice(len(X_train_balanced), len(X_train_balanced), replace=True)
                X_boot = X_train_balanced[indices]
                y_boot = y_train_balanced[indices]
                
                # Train a new random forest
                rf_boot = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=i
                )
                rf_boot.fit(X_boot, y_boot)
                importances[i, :] = rf_boot.feature_importances_
            
            # Calculate mean and confidence intervals
            importance_mean = np.mean(importances, axis=0)
            importance_std = np.std(importances, axis=0)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_mean,
                'std': importance_std
            }).sort_values('importance', ascending=False)
            
            logging.info("\nTop 10 Most Important Features (with 95% confidence intervals):")
            for _, row in feature_importance.head(10).iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.4f} ± {1.96*row['std']:.4f}")
            
            # Save feature importance
            feature_importance.to_csv(self.models_dir / "feature_importance.csv", index=False)
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = MicrobiomeModelTrainer()
    trainer.train_models() 