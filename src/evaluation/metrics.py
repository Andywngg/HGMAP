def comprehensive_evaluation(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': balanced_accuracy_score(y_true, y_pred),
        'auroc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'calibration': calibration_score(y_true, y_prob)
    }
    
    # Statistical significance tests
    metrics['p_value'] = mcnemar_test(y_true, y_pred)
    
    return metrics 