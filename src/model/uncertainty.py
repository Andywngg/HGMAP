import torch
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List

class UncertaintyQuantifier:
    def __init__(
        self, 
        model,
        n_samples: int = 50,
        dropout_rate: float = 0.3,
        temperature: float = 1.0
    ):
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        
    def enable_dropout(self):
        """Enable dropout during inference"""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
                m.p = self.dropout_rate
    
    def monte_carlo_dropout(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Perform MC Dropout for uncertainty estimation"""
        self.enable_dropout()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X)
                pred = torch.softmax(pred / self.temperature, dim=1)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.var(predictions, axis=0)
        aleatoric_uncertainty = np.mean(predictions * (1 - predictions), axis=0)
        
        return mean_pred, {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': epistemic_uncertainty + aleatoric_uncertainty
        }
    
    def confidence_intervals(
        self, 
        predictions: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        upper = np.percentile(predictions, (1-alpha/2) * 100, axis=0)
        
        return {
            'lower_bound': lower,
            'upper_bound': upper,
            'confidence_width': upper - lower
        }
    
    def entropy_uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate predictive entropy"""
        return -np.sum(predictions * np.log(predictions + 1e-10), axis=-1)
    
    def mutual_information(
        self, 
        predictions: np.ndarray
    ) -> np.ndarray:
        """Calculate mutual information (epistemic uncertainty)"""
        expected_entropy = -np.mean(
            np.sum(predictions * np.log(predictions + 1e-10), axis=-1),
            axis=0
        )
        entropy_expected = -np.sum(
            np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + 1e-10)
        )
        return entropy_expected - expected_entropy
    
    def calibration_metrics(
        self, 
        predictions: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration metrics"""
        confidences = np.max(predictions, axis=-1)
        pred_labels = np.argmax(predictions, axis=-1)
        
        # Expected Calibration Error
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(
                confidences > bin_lower,
                confidences <= bin_upper
            )
            
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(pred_labels[in_bin] == true_labels[in_bin])
                confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * np.mean(in_bin)
        
        return {
            'ece': float(ece),
            'mean_confidence': float(np.mean(confidences)),
            'confidence_std': float(np.std(confidences))
        } 