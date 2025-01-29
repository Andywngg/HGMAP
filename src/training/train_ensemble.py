import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from ..data.processor_final import MicrobiomeProcessor

class EnsembleTrainer:
    def __init__(
        self,
        data_dir: str = "data",
        n_splits: int = 10,
        random_state: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.processor = MicrobiomeProcessor(data_dir=data_dir)
        
        # Initialize base models
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'xgb': xgb.XGBClassifier(n_estimators=100, random_state=random_state),
            'lgb': lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        }
        
        # Initialize stacking classifier
        self.stacking = StackingClassifier(
            estimators=[
                ('rf', self.base_models['rf']),
                ('gb', self.base_models['gb']),
                ('xgb', self.base_models['xgb']),
                ('lgb', self.base_models['lgb'])
            ],
            final_estimator=lgb.LGBMClassifier(n_estimators=100),
            cv=5,
            n_jobs=-1
        )
    
    def evaluate_model(self, model, X, y, model_name=""):
        """Evaluate model performance with cross-validation."""
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            metrics['precision'].append(precision_score(y_val, y_pred))
            metrics['recall'].append(recall_score(y_val, y_pred))
            metrics['f1'].append(f1_score(y_val, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_val, y_prob))
        
        # Log results
        self.logger.info(f"\nResults for {model_name}:")
        for metric, values in metrics.items():
            self.logger.info(f"{metric}: {np.mean(values):.3f} (+/- {np.std(values):.3f})")
        
        return metrics
    
    def generate_shap_analysis(self, model, X, output_dir="reports/figures"):
        """Generate SHAP analysis plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png")
        plt.close()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_importance.png")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, output_dir="reports/figures"):
        """Plot confusion matrix."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        plt.close()
    
    def train_and_evaluate(self):
        """Train and evaluate all models."""
        try:
            # Load and preprocess data
            self.logger.info("Loading and preprocessing data...")
            X, y = self.processor.prepare_data()
            
            # Evaluate base models
            base_metrics = {}
            for name, model in self.base_models.items():
                self.logger.info(f"\nEvaluating {name}...")
                metrics = self.evaluate_model(model, X, y, name)
                base_metrics[name] = metrics
            
            # Evaluate stacking ensemble
            self.logger.info("\nEvaluating stacking ensemble...")
            ensemble_metrics = self.evaluate_model(self.stacking, X, y, "Stacking Ensemble")
            
            # Train final model on full dataset
            self.logger.info("\nTraining final ensemble model...")
            self.stacking.fit(X, y)
            
            # Generate SHAP analysis
            self.logger.info("\nGenerating SHAP analysis...")
            self.generate_shap_analysis(self.stacking, X)
            
            # Generate confusion matrix
            y_pred = self.stacking.predict(X)
            self.plot_confusion_matrix(y, y_pred)
            
            # Save classification report
            report = classification_report(y, y_pred)
            with open("reports/classification_report.txt", "w") as f:
                f.write(report)
            
            # Save model metrics
            metrics_df = pd.DataFrame({
                **{f"{name}_{k}": np.mean(v) for name, metrics in base_metrics.items() 
                   for k, v in metrics.items()},
                **{f"ensemble_{k}": np.mean(v) for k, v in ensemble_metrics.items()}
            }, index=[0])
            
            metrics_df.to_csv("reports/model_metrics.csv", index=False)
            
            return self.stacking
            
        except Exception as e:
            self.logger.error(f"Error in train_and_evaluate: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = EnsembleTrainer()
    model = trainer.train_and_evaluate() 