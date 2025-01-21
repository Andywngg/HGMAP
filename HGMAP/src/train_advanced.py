import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.advanced_preprocessing import AdvancedPreprocessor
from src.features.microbiome_features import MicrobiomeFeatureEngineer
from src.model.losses import MicrobiomeLoss
from src.model.transfer import TransferLearningModule
from src.optimization.hyperopt import AdvancedHyperOptimizer
from src.model.advanced_ensemble import HyperEnsemble

class AdvancedTrainingPipeline:
    def __init__(
        self,
        data_path: str,
        source_data_paths: List[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.data_path = data_path
        self.source_data_paths = source_data_paths or []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data"""
        # Load main dataset
        data = pd.read_excel(self.data_path)
        X = data.drop(columns=['target']).values
        y = data['target'].values
        
        # Load source datasets for transfer learning
        source_datasets = []
        for path in self.source_data_paths:
            source_data = pd.read_excel(path)
            source_datasets.append({
                'X': source_data.drop(columns=['target']).values,
                'y': source_data['target'].values
            })
        
        # Advanced preprocessing
        preprocessor = AdvancedPreprocessor(
            n_components=50,
            use_umap=True,
            use_pca=True,
            use_kernel_pca=True
        )
        X_preprocessed = preprocessor.fit_transform(X)
        
        # Feature engineering
        feature_engineer = MicrobiomeFeatureEngineer(
            min_prevalence=0.1,
            min_abundance=0.001,
            n_clusters=20
        )
        X_engineered = feature_engineer.fit_transform(
            X_preprocessed,
            feature_names=data.columns[:-1].tolist()
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'test': (X_test, y_test),
            'source_datasets': source_datasets
        }
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize model hyperparameters"""
        optimizer = AdvancedHyperOptimizer(
            model_builder=self.build_model,
            n_trials=100,
            cv_folds=5,
            metric='auc'
        )
        
        best_params, best_score = optimizer.optimize(X_train, y_train)
        print(f"Best validation score: {best_score}")
        return best_params
    
    def build_model(self, **params):
        """Build model with given parameters"""
        model = HyperEnsemble(**params)
        return model
    
    def train_model(self, data_dict, hyperparameters):
        """Train the model with all improvements"""
        X_train, y_train = data_dict['train']
        X_test, y_test = data_dict['test']
        
        # Initialize model with best hyperparameters
        model = self.build_model(**hyperparameters)
        
        # Initialize transfer learning
        transfer_module = TransferLearningModule(
            base_model=model,
            source_datasets=data_dict['source_datasets'],
            n_epochs=50,
            lr=1e-4
        )
        
        # Apply transfer learning
        model = transfer_module.apply_transfer_learning(
            target_data={'X': X_train, 'y': y_train}
        )
        
        # Initialize custom loss
        criterion = MicrobiomeLoss(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            focal_gamma=2.0,
            class_weights=torch.tensor(
                [1.0, (y_train == 0).sum() / (y_train == 1).sum()]
            ).to(self.device)
        )
        
        # Train with custom loss
        model.train_with_custom_loss(
            X_train, y_train,
            criterion=criterion,
            n_epochs=100,
            batch_size=32,
            learning_rate=1e-4
        )
        
        return model, (X_test, y_test)
    
    def evaluate_model(self, model, test_data):
        """Evaluate the trained model"""
        X_test, y_test = test_data
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_prob[:, 1]),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data_dict = self.load_and_preprocess_data()
        
        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = self.optimize_hyperparameters(
            data_dict['train'][0],
            data_dict['train'][1]
        )
        
        # Train model
        print("Training model with all improvements...")
        model, test_data = self.train_model(data_dict, best_params)
        
        # Evaluate
        print("Evaluating model...")
        metrics = self.evaluate_model(model, test_data)
        
        print("\nFinal Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return model, metrics

# Usage
if __name__ == "__main__":
    pipeline = AdvancedTrainingPipeline(
        data_path="path/to/your/main/data.xlsx",
        source_data_paths=[
            "path/to/source/data1.xlsx",
            "path/to/source/data2.xlsx"
        ]
    )
    
    model, metrics = pipeline.run_pipeline() 