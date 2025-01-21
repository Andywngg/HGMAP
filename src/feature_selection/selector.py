import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import shap
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_features=100,
        shap_threshold=0.01,
        mi_percentile=75,
        random_state=42
    ):
        self.n_features = n_features
        self.shap_threshold = shap_threshold
        self.mi_percentile = mi_percentile
        self.random_state = random_state
        self.selected_features_ = None
        
    def fit(self, X, y):
        # Initialize base model for SHAP
        base_model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        base_model.fit(X, y)
        
        # SHAP-based selection
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X)
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_selected = shap_importance > np.percentile(shap_importance, self.shap_threshold * 100)
        
        # Mutual Information selection
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_selected = mi_scores > np.percentile(mi_scores, self.mi_percentile)
        
        # Boruta selection
        boruta = BorutaPy(
            estimator=RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            n_estimators='auto',
            random_state=self.random_state
        )
        boruta.fit(X, y)
        
        # Combine selections using voting
        combined_votes = shap_selected.astype(int) + mi_selected.astype(int) + boruta.support_
        self.selected_features_ = np.where(combined_votes >= 2)[0]
        
        # Limit to top n_features if needed
        if len(self.selected_features_) > self.n_features:
            importance_scores = shap_importance + mi_scores + boruta.ranking_
            top_indices = np.argsort(importance_scores[self.selected_features_])[-self.n_features:]
            self.selected_features_ = self.selected_features_[top_indices]
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_features_] 