import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

class AdvancedCalibrator:
    def __init__(self, base_model, method='hybrid'):
        self.base_model = base_model
        self.method = method
        self.calibrators = []
        
    def fit(self, X, y):
        # Split data for calibration
        n_splits = 5
        fold_size = len(X) // n_splits
        
        for i in range(n_splits):
            # Create fold indices
            val_idx = slice(i * fold_size, (i + 1) * fold_size)
            train_idx = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, len(X)))
            
            # Train base model
            self.base_model.fit(X[train_idx], y[train_idx])
            
            # Get predictions for calibration
            probs = self.base_model.predict_proba(X[val_idx])
            
            # Train calibrators
            if self.method == 'hybrid':
                isotonic = IsotonicRegression(out_of_bounds='clip')
                isotonic.fit(probs[:, 1], y[val_idx])
                
                platt = CalibratedClassifierCV(
                    base_estimator=self.base_model,
                    cv='prefit',
                    method='sigmoid'
                )
                platt.fit(X[val_idx], y[val_idx])
                
                self.calibrators.append({
                    'isotonic': isotonic,
                    'platt': platt,
                    'weight': None  # Will be set during prediction
                })
        
        return self
    
    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)
        
        calibrated_probs = []
        for calibrator in self.calibrators:
            # Get calibrated probabilities from both methods
            isotonic_probs = calibrator['isotonic'].predict(base_probs[:, 1])
            platt_probs = calibrator['platt'].predict_proba(X)[:, 1]
            
            # Compute weights based on confidence
            isotonic_conf = np.abs(isotonic_probs - 0.5)
            platt_conf = np.abs(platt_probs - 0.5)
            
            # Weight predictions based on confidence
            total_conf = isotonic_conf + platt_conf
            isotonic_weight = isotonic_conf / total_conf
            platt_weight = platt_conf / total_conf
            
            # Combine predictions
            combined_probs = (isotonic_probs * isotonic_weight + 
                            platt_probs * platt_weight)
            
            calibrated_probs.append(combined_probs)
        
        # Average across all calibrators
        final_probs = np.mean(calibrated_probs, axis=0)
        return np.column_stack((1 - final_probs, final_probs)) 