import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Callable

class AdvancedHyperOptimizer:
    def __init__(
        self,
        model_builder: Callable,
        n_trials: int = 100,
        cv_folds: int = 5,
        metric: str = 'auc',
        direction: str = 'maximize'
    ):
        self.model_builder = model_builder
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.direction = direction
        
        # Initialize optuna study with advanced settings
        self.study = optuna.create_study(
            sampler=TPESampler(
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True
            ),
            pruner=SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0
            ),
            direction=direction
        )
        
    def _get_param_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define hyperparameter search space"""
        params = {
            # Network Architecture
            'n_layers': trial.suggest_int('n_layers', 2, 5),
            'hidden_dims': [
                trial.suggest_int(f'hidden_dim_{i}', 32, 512)
                for i in range(trial.suggest_int('n_layers', 2, 5))
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'attention_heads': trial.suggest_int('attention_heads', 4, 16),
            
            # Training Parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            
            # Advanced Parameters
            'attention_dropout': trial.suggest_float('attention_dropout', 0.1, 0.4),
            'layer_norm_eps': trial.suggest_loguniform('layer_norm_eps', 1e-7, 1e-5),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.1, 1.0),
            
            # Ensemble Parameters
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # Conditional parameters
        if trial.suggest_categorical('use_residual', [True, False]):
            params['residual_dropout'] = trial.suggest_float('residual_dropout', 0.1, 0.4)
        
        return params
    
    def optimize(self, X: np.ndarray, y: np.ndarray):
        """Run hyperparameter optimization"""
        def objective(trial):
            params = self._get_param_space(trial)
            cv_scores = []
            
            # Stratified K-Fold Cross-validation
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                model = self.model_builder(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                # Evaluate
                if self.metric == 'auc':
                    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                else:
                    score = accuracy_score(y_val, model.predict(X_val))
                
                cv_scores.append(score)
                
                # Report intermediate value for pruning
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(cv_scores)
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=None,  # No timeout
            n_jobs=-1  # Use all available cores
        )
        
        return self.study.best_params, self.study.best_value 