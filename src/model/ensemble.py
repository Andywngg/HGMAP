class AdvancedEnsemble:
    def __init__(self):
        self.models = {
            'gnn': AdvancedGraphNeuralNetwork(),
            'transformer': MicrobiomeTransformer(),
            'xgboost': XGBClassifier(),
            'lightgbm': LGBMClassifier()
        }
        
    def weighted_prediction(self, X):
        predictions = {}
        weights = self._compute_dynamic_weights(X)
        
        for name, model in self.models.items():
            pred = model.predict_proba(X)
            predictions[name] = pred * weights[name]
        
        return np.sum(list(predictions.values()), axis=0) 