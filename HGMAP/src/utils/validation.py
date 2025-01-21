def validate_input_data(X, y=None):
    """Validate input data format and contents"""
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert not np.isnan(X).any(), "Input contains NaN values"
    assert not np.isinf(X).any(), "Input contains infinite values"
    
    if y is not None:
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert len(y) == len(X), "X and y must have same length"
        assert len(y.shape) == 1, "y must be 1-dimensional"
        assert len(np.unique(y)) == 2, "y must be binary"

def validate_model_output(y_pred):
    """Validate model predictions"""
    assert isinstance(y_pred, np.ndarray), "Predictions must be a numpy array"
    assert not np.isnan(y_pred).any(), "Predictions contain NaN values"
    assert not np.isinf(y_pred).any(), "Predictions contain infinite values"
    if len(y_pred.shape) == 2:
        assert y_pred.shape[1] == 2, "Binary classification requires 2 columns"
        assert np.allclose(np.sum(y_pred, axis=1), 1), "Probabilities must sum to 1" 