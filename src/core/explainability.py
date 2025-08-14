import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance



def get_feature_importance(model_name, model, X_data, y_data, random_state=42, use_permutation_importance=False):
    """Get feature importance for a given model.

    Args:
        model_name (str): The name of the model.
        model (sklearn.base.BaseEstimator): The trained model.
        X_data (pd.DataFrame): The feature data.
        y_data (pd.Series): The target data.
        random_state (int): The random state to use for the permutation importance.

    Returns:
        pd.DataFrame: A dataframe with the feature importance.
    """
    classifier = model.best_estimator_.named_steps['classifier']
    feature_names = X_data.columns
    
    if hasattr(classifier, 'feature_importances_') and not use_permutation_importance:
        # Tree-based models (Random Forest, XGBoost)
        print(f"Using feature_importances_ for {model_name}...")
        importance_values = classifier.feature_importances_
        importance_type = 'Built-in'
        
    elif hasattr(classifier, 'coef_') and not use_permutation_importance:
        # Linear models (Logistic Regression) or SVM with linear kernel
        print(f"Using coef_ for {model_name}...")
        importance_values = np.abs(classifier.coef_[0])  # Use absolute values of coefficients
        importance_type = 'Coefficient Magnitude'
        
    else:
        # Models without built-in importance (SVM with RBF kernel)
        # or if use_permutation_importance is True
        # TODO: Should we use a different approach: SHAP or LOFO?
        print(f"Computing permutation importance for {model_name}...")
        perm_importance = permutation_importance(
            model, X_data, y_data, 
            n_repeats=10, 
            random_state=random_state,
            scoring='roc_auc'
        )
        importance_values = perm_importance.importances_mean
        importance_type = 'Permutation'
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
        'importance_type': importance_type
    }).sort_values('importance', ascending=False)
    
    return importance_df
